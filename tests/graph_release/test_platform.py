"""Pins the builder's reads and platform parts on inputs recorded from dev Spanner.

`raw_inputs.json` holds one snapshot's reads for twenty variables across every provenance; its
`source` field names where, when and how the reads were scoped. It carries a second-hop edge the
closure keeps (`..._CyclicallyAdjusted_RealValue measurementQualifier RealValue`, whose object a
live variable's edge names) and ones it drops (`ConversionRecipeDAC isProvisional True`, whose
object has no `Node` row).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from artifact_publication.jobs import ImportKey
from graph_release import reads
from graph_release.authored import AUTHORED, AUTHORED_PROVENANCE, literal_id
from graph_release.format import EDGES, FACETS, IMPORTS, NODES, SERIES
from graph_release.platform import (
    PlatformParts,
    PlatformRefusal,
    build_platform_parts,
    stage_platform_parts,
)
from graph_release.reads import RawInputs, read_inputs, snapshot_at

_FIXTURE = Path(__file__).parents[1] / "fixtures/graph_release/raw_inputs/raw_inputs.json"

_KEPT = (
    "ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_"
    "CyclicallyAdjusted_RealValue",
    "measurementQualifier",
    "RealValue",
    "provenance/IMFWorldEconomicOutlook",
)
_DROPPED = ("ConversionRecipeDAC", "isProvisional", "True", "provenance/DFOOntology")
_SKILL_PREDICATES = frozenset(
    {"typeOf", "skillAppliesTo", "skillAppliesToProvenance", "agentSkillReferences", "url"}
)

_NOW = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
_HISTORY = ["wf-1", "SUCCESS", False, _NOW, _NOW]
_HISTORY_COLUMNS = (
    "WorkflowExecutionID",
    "Status",
    "IngestionFailure",
    "CreationTimestamp",
    "CompletionTimestamp",
)
_HISTORY_ROW = dict(zip(_HISTORY_COLUMNS, _HISTORY, strict=True))


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(_FIXTURE.read_text())


def _table(recorded: dict, name: str, schema: pa.Schema) -> pa.Table:
    table = recorded["tables"][name]
    assert table["columns"] == schema.names
    columns = list(zip(*table["rows"], strict=True))
    arrays = []
    for field, values in zip(schema, columns, strict=True):
        if pa.types.is_timestamp(field.type):
            values = [None if v is None else datetime.fromisoformat(v) for v in values]
        arrays.append(pa.array(list(values), field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


@pytest.fixture
def raw(recorded: dict) -> RawInputs:
    return RawInputs(
        import_key=ImportKey.from_rows(None, _HISTORY_ROW, _HISTORY_ROW),
        series_summary=_table(recorded, "series_summary", reads.RAW_SERIES),
        facets=_table(recorded, "facets", reads.RAW_FACETS),
        live_variables=_table(recorded, "live_variables", reads.RAW_LIVE_VARIABLES),
        edges_raw=_table(recorded, "edges_raw", reads.RAW_EDGES),
        nodes_raw=_table(recorded, "nodes_raw", reads.RAW_NODES),
        imports=_table(recorded, "imports", reads.RAW_IMPORTS),
    )


def _edge_set(table: pa.Table) -> set[tuple[str, ...]]:
    return {tuple(row.values()) for row in table.cast(reads.RAW_EDGES).to_pylist()}


# ---------------------------------------------------------------- reads


class _RecordedSpanner:
    """Answers each statement, by its request tag, from the recorded tables as Spanner would:
    edge reads filter the recorded edges by their parameters."""

    def __init__(self, raw: RawInputs) -> None:
        self.raw = raw
        self.calls: list[tuple[str, dict | None]] = []
        edges = raw.edges_raw.to_pylist()
        self.edges = [tuple(edge.values()) for edge in edges]
        self.typed = {node["subject_id"] for node in raw.nodes_raw.to_pylist() if node["types"]}

    def execute_sql(self, sql, params=None, param_types=None, *, request_options=None):
        tag = request_options["request_tag"]
        assert tag.startswith("app=graph-release,action=")
        action = tag.removeprefix("app=graph-release,action=")
        self.calls.append((action, params))
        if params is not None:
            assert set(param_types) == set(params)
        if action == "import_key":
            return [[None]] if "IngestionLock" in sql else [_HISTORY]
        if action == "variable_edges":
            subjects = set(params["subjects"])
            return [(*e, e[2] in self.typed) for e in self.edges if e[0] in subjects]
        if action == "code_edges":
            members = {e[0] for e in self.edges if e[1] == "typeOf" and e[2] in params["enums"]}
            return [e for e in self.edges if e[0] in members]
        if action == "skill_edges":
            skills = {e[0] for e in self.edges if e[1] == "typeOf" and e[2] == "AgentSkill"}
            return [e for e in self.edges if e[0] in skills and e[1] in _SKILL_PREDICATES]
        if action == "specialization_parents":
            codes = set(params["codes"])
            return [
                (e[0], e[2]) for e in self.edges if e[1] == "specializationOf" and e[0] in codes
            ]
        if action == "target_edges":
            subjects = set(params["subjects"])
            return [e for e in self.edges if e[0] in subjects]
        if action == "nodes":
            ids = set(params["ids"])
            return [
                list(n.values()) for n in self.raw.nodes_raw.to_pylist() if n["subject_id"] in ids
            ]
        table = {
            "live_variables": self.raw.live_variables,
            "facets": self.raw.facets,
            "series": self.raw.series_summary,
            "imports": self.raw.imports,
        }[action]
        return [list(row.values()) for row in table.to_pylist()]


def test_read_inputs_runs_every_read_tagged_in_one_snapshot(raw, monkeypatch):
    monkeypatch.setattr(reads, "CHUNK_ROWS", 7)
    spanner = _RecordedSpanner(raw)

    read = read_inputs(spanner)

    actions = [action for action, _ in spanner.calls]
    assert actions[:3] == ["import_key"] * 3
    assert sorted(set(actions[3:])) == sorted(
        [
            "live_variables",
            "variable_edges",
            "code_edges",
            "skill_edges",
            "specialization_parents",
            "target_edges",
            "nodes",
            "facets",
            "series",
            "imports",
        ]
    )
    assert read.import_key == raw.import_key
    assert _edge_set(read.edges_raw) == _edge_set(raw.edges_raw)
    assert read.edges_raw.num_rows == len(_edge_set(raw.edges_raw))
    for name in ("series_summary", "facets", "live_variables", "nodes_raw", "imports"):
        got, want = getattr(read, name), getattr(raw, name)
        assert got.schema == want.schema
        assert got.to_pylist() == want.to_pylist()
    assert read.series_summary["variable_measured"].num_chunks > 1


def test_read_inputs_walks_the_specialization_chain_to_its_top_and_stops_at_a_cycle(raw):
    edges = raw.edges_raw.to_pylist()
    live = set(raw.live_variables["variable_measured"].to_pylist())
    constrained = {
        (e["subject_id"], e["object_id"]) for e in edges if e["predicate"] == "constraintProperties"
    }
    value = next(
        e["object_id"]
        for e in edges
        if e["subject_id"] in live and (e["subject_id"], e["predicate"]) in constrained
    )
    parents = {
        e["subject_id"]: e["object_id"] for e in edges if e["predicate"] == "specializationOf"
    }
    top = value
    while top in parents:
        top = parents[top]
    # Ten levels above the chain's top, the last pointing back at the third.
    chain = [top, *(f"dfo/Ancestor{level}" for level in range(1, 11))]
    added = [
        (child, "specializationOf", parent, "provenance/DFOOntology")
        for child, parent in [*pairwise(chain), (chain[-1], chain[3])]
    ]
    spanner = _RecordedSpanner(
        dataclasses.replace(raw, edges_raw=pa.concat_tables([raw.edges_raw, _edges(added)]))
    )

    read = read_inputs(spanner)

    assert set(added) <= _edge_set(read.edges_raw)
    levels = [
        params["codes"] for action, params in spanner.calls if action == "specialization_parents"
    ]
    assert levels[-1] == [chain[-1]]
    assert sum(code == chain[3] for level in levels for code in level) == 1


def test_snapshot_at_refuses_a_naive_timestamp():
    class Database:
        def snapshot(self, **kwargs):
            return kwargs

    assert snapshot_at(Database(), read_timestamp=_NOW) == {
        "read_timestamp": _NOW,
        "multi_use": True,
    }
    with pytest.raises(ValueError, match="timezone-aware"):
        snapshot_at(Database(), read_timestamp=_NOW.replace(tzinfo=None))


# ---------------------------------------------------------------- platform parts


def test_platform_parts_hold_the_format_schemas_in_key_order(raw):
    parts = build_platform_parts(raw, authored=AUTHORED)

    expected = {
        "series": SERIES.schema.remove(SERIES.schema.get_field_index("entity3")),
        "facets": FACETS.schema,
        "edges": EDGES.schema,
        "nodes": NODES.schema,
        "imports": IMPORTS.schema,
    }
    keys = {"series": SERIES.key, "facets": FACETS.key, "edges": EDGES.key, "nodes": NODES.key}
    for name, table in parts.tables().items():
        assert table.schema == expected[name], name
        if name in keys:
            rows = [tuple(row[k] for k in keys[name]) for row in table.to_pylist()]
            assert rows == sorted(rows), name
    assert parts.series.num_rows == raw.series_summary.num_rows
    assert parts.facets.num_rows == raw.facets.num_rows


def test_series_masks_count_every_observation(raw):
    series = build_platform_parts(raw, authored=AUTHORED).series

    words = [series[f"period_mask_{w}"].to_numpy().view(np.uint64) for w in range(2)]
    popcount = sum(np.bitwise_count(word).astype(np.int64) for word in words)
    assert (popcount == series["observation_count"].to_numpy()).all()
    first = series.slice(0, 1).to_pylist()[0]
    earliest = raw.series_summary.filter(
        pc.and_(
            pc.equal(
                raw.series_summary["variable_measured"].cast(pa.string()),
                first["variable_measured"],
            ),
            pc.equal(raw.series_summary["entity1"].cast(pa.string()), first["entity1"]),
        )
    )["earliest_date"][0].as_py()
    assert first["earliest_date"] == date(int(earliest), 1, 1)


def test_edge_closure_keeps_named_second_hop_targets_and_drops_the_rest(raw):
    parts = build_platform_parts(raw, authored=AUTHORED)

    raw_edges, edges = _edge_set(raw.edges_raw), _edge_set(parts.edges)
    assert {_KEPT, _DROPPED} <= raw_edges, "the fixture must hold both closure cases"
    assert _KEPT in edges
    assert _DROPPED not in edges
    nodes = set(parts.nodes["subject_id"].to_pylist())
    assert {end for edge in edges for end in (edge[0], edge[2])} <= nodes
    live = set(raw.live_variables["variable_measured"].to_pylist())
    first_hop = {edge for edge in raw_edges if edge[0] in live}
    assert first_hop <= edges


def test_authored_rows_join_nodes_and_edges(raw):
    parts = build_platform_parts(raw, authored=AUTHORED)

    authored = parts.nodes.filter(parts.nodes["authored"])
    assert sorted(authored["subject_id"].to_pylist()) == sorted(
        n.subject_id for n in AUTHORED.nodes
    )
    assert len(AUTHORED.nodes) == 15
    authored_edges = {e for e in _edge_set(parts.edges) if e[3] == AUTHORED_PROVENANCE}
    assert authored_edges == {
        (e.subject_id, e.predicate, e.object_id, AUTHORED_PROVENANCE) for e in AUTHORED.edges
    }
    assert len(authored_edges) == 4


def test_literal_id_is_the_id_spanner_gives_a_literal(raw):
    literals = [n for n in raw.nodes_raw.to_pylist() if n["value"] and not n["types"]]
    assert literals
    for node in literals:
        assert literal_id(node["value"]) == node["subject_id"]


def test_stage_one_drops_the_predicates_the_mixer_blocks(raw):
    variable = raw.live_variables["variable_measured"][0].as_py()
    blocked = (variable, "linkedMemberOf", "dc/g/Root", "provenance/DFOOntology")
    edges = pa.concat_tables([raw.edges_raw, _edges([blocked])])

    parts = build_platform_parts(dataclasses.replace(raw, edges_raw=edges), authored=AUTHORED)

    assert blocked not in _edge_set(parts.edges)


def _edges(rows: list[tuple[str, ...]]) -> pa.Table:
    return pa.Table.from_pylist(
        [dict(zip(reads.RAW_EDGES.names, row, strict=True)) for row in rows],
        schema=reads.RAW_EDGES,
    )


def _with_facet(raw: RawInputs, facet: dict | None, *, other: dict | None = None) -> RawInputs:
    facets = raw.facets.to_pylist()
    facets[0] = {
        "facet_id": facets[0]["facet_id"],
        "facet_min": json.dumps(facet),
        "facet_max": json.dumps(other or facet),
    }
    return dataclasses.replace(raw, facets=pa.Table.from_pylist(facets, schema=reads.RAW_FACETS))


_FACET = {"isDcAggregate": False, "provenance": "provenance/OECDDAC1", "unit": "CurrencyUSD"}


def _role_counts(raw: RawInputs) -> dict[str, int]:
    """Each variable's count of declared `observationProperties` roles."""
    roles: dict[str, set[str]] = {}
    for edge in raw.edges_raw.to_pylist():
        if edge["predicate"] == "observationProperties":
            roles.setdefault(edge["subject_id"], set()).add(edge["object_id"])
    return {variable: len(found) for variable, found in roles.items()}


def _with_series_column(
    raw: RawInputs, name: str, value: object, *, variables: set[str] | None = None
) -> RawInputs:
    """`raw` with `name` set to `value` on the series of `variables`, every series when
    `None`."""
    summary = raw.series_summary
    rows = summary.to_pylist()
    for row in rows:
        if variables is None or row["variable_measured"] in variables:
            row[name] = value(row[name]) if callable(value) else value
    return dataclasses.replace(
        raw, series_summary=pa.Table.from_pylist(rows, schema=summary.schema)
    )


@pytest.mark.parametrize(
    ("break_inputs", "code"),
    [
        (
            lambda raw: _with_facet(raw, _FACET | {"observationPeriod": "P1M"}),
            "unsupported_observation_period",
        ),
        (
            lambda raw: _with_facet(
                raw,
                _FACET | {"observationPeriod": "P1Y"},
                other=_FACET | {"observationPeriod": "P1Y", "unit": "CurrencyEUR"},
            ),
            "inconsistent_facet",
        ),
        (lambda raw: _with_facet(raw, None), "inconsistent_facet"),
        (lambda raw: _with_facet(raw, {"observationPeriod": "P1Y"}), "inconsistent_facet"),
        (lambda raw: dataclasses.replace(raw, facets=raw.facets.slice(1)), "unknown_facet"),
        (
            lambda raw: _with_series_column(raw, "mask_years", lambda years: years + 1),
            "popcount_mismatch",
        ),
        (
            lambda raw: _with_series_column(
                raw,
                "outside_mask",
                True,
                variables={v for v, count in _role_counts(raw).items() if count <= 2},
            ),
            "unsupported_observation_period",
        ),
        (
            lambda raw: _with_series_column(
                raw,
                "observation_count",
                lambda count: count + 1,
                variables={v for v, count in _role_counts(raw).items() if count <= 2},
            ),
            "popcount_mismatch",
        ),
        (
            lambda raw: dataclasses.replace(
                raw,
                nodes_raw=pa.concat_tables(
                    [
                        raw.nodes_raw,
                        pa.Table.from_pylist(
                            [
                                {
                                    "subject_id": "Nominal",
                                    "value": None,
                                    "name": "Nominal",
                                    "types": [],
                                }
                            ],
                            schema=reads.RAW_NODES,
                        ),
                    ]
                ),
            ),
            "authored_mismatch",
        ),
        (
            lambda raw: dataclasses.replace(
                raw,
                edges_raw=pa.concat_tables(
                    [
                        raw.edges_raw,
                        _edges(
                            [
                                (
                                    raw.live_variables["variable_measured"][0].as_py(),
                                    "measuredProperty",
                                    "noSuchNode",
                                    "provenance/OECDDAC1",
                                )
                            ]
                        ),
                    ]
                ),
            ),
            "dangling_edge",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_platform_refuses_inputs_the_release_cannot_hold(raw, break_inputs, code):
    build_platform_parts(raw, authored=AUTHORED)

    with pytest.raises(PlatformRefusal) as refused:
        build_platform_parts(break_inputs(raw), authored=AUTHORED)

    assert refused.value.code == code


def test_a_series_with_no_roles_keeps_years_outside_the_mask_and_same_year_dates(raw):
    """Its dates and `observation_count` hold the observations; its mask and the part's columns
    do not change."""
    declared = _role_counts(raw)
    live = set(raw.live_variables["variable_measured"].to_pylist())
    first, *rest = sorted(live - {v for v, count in declared.items() if count <= 2})
    assert rest, "the fixture must hold two variables declaring no role"
    # `first` has a year outside the mask, the rest a second date in one year.
    unmasked = _with_series_column(raw, "outside_mask", True, variables={first})
    unmasked = _with_series_column(
        unmasked, "observation_count", lambda count: count + 1, variables={first, *rest}
    )

    parts = build_platform_parts(unmasked, authored=AUTHORED)

    assert parts.series.schema == build_platform_parts(raw, authored=AUTHORED).series.schema
    assert "outside_mask" not in parts.series.column_names
    assert parts.series.num_rows == raw.series_summary.num_rows


def test_observation_dates_parse_years_and_months_and_refuse_other_forms(raw):
    summary = raw.series_summary
    months = summary.set_column(
        summary.schema.get_field_index("earliest_date"),
        "earliest_date",
        pa.array(["2001-06"] * summary.num_rows),
    )
    parts = build_platform_parts(dataclasses.replace(raw, series_summary=months), authored=AUTHORED)
    assert set(parts.series["earliest_date"].to_pylist()) == {date(2001, 6, 1)}

    quarters = months.set_column(
        months.schema.get_field_index("earliest_date"),
        "earliest_date",
        pa.array(["2001-Q1"] * summary.num_rows),
    )
    with pytest.raises(PlatformRefusal, match="2001-Q1") as refused:
        build_platform_parts(dataclasses.replace(raw, series_summary=quarters), authored=AUTHORED)
    assert refused.value.code == "unsupported_observation_period"


# ---------------------------------------------------------------- staging


def _staged_tables(
    parts: PlatformParts, tmp_path: Path
) -> dict[str, list[tuple[object, pa.Table]]]:
    staged: dict[str, list[tuple[object, pa.Table]]] = {}
    for part in stage_platform_parts(parts, staging_dir=tmp_path, execution_id="exec-1"):
        assert part.ref.produced_by_execution == "exec-1"
        table = pa.ipc.open_file(pa.memory_map(str(part.path), "r")).read_all()
        staged.setdefault(part.ref.name, []).append((part.ref, table))
    return staged


def test_staged_files_hold_every_row_split_by_provenance(raw, tmp_path):
    parts = build_platform_parts(raw, authored=AUTHORED)

    staged = _staged_tables(parts, tmp_path)

    assert set(staged) == set(parts.tables())
    for name, table in parts.tables().items():
        rows = [row for _, file in staged[name] for row in file.to_pylist()]
        assert sorted(map(str, rows)) == sorted(map(str, table.to_pylist())), name
    facet_provenance = dict(
        zip(
            parts.facets["facet_id"].to_pylist(),
            parts.facets["provenance"].to_pylist(),
            strict=True,
        )
    )
    for ref, file in staged["series"]:
        assert {facet_provenance[f] for f in file["facet_id"].to_pylist()} == {ref.provenance}
    for ref, file in staged["edges"]:
        assert set(file["provenance"].to_pylist()) == {ref.provenance}
        expected = (
            "graph_release.authored" if ref.provenance == AUTHORED_PROVENANCE else "spanner.edge"
        )
        assert ref.source == expected
