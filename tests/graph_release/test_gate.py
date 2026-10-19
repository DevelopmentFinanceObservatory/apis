"""Pins the release gate and the shadow diff on releases built from recorded dev data.

The invariant, SDMX and coverage checks run on the platform parts of `raw_inputs.json` (see
`test_platform.py`). The mixer differential and the origin arc diff run on
`dc_graph/visible_arcs/ids_variables.json`: five IDS variables' Spanner `Edge` rows and the dev
mixer's `->*` answer for them, whose `constraintProperties` carry the import-added
`sourceIndicatorCode`. Each check first passes on an unbroken release, then fails on one built
to break it.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from google.auth.exceptions import RefreshError

from dc_graph import GraphClient
from dc_graph.errors import GraphReadError
from dc_graph.visible_arcs import IMPORT_ADDED_CONSTRAINT_PROPERTIES
from dc_sdmx.errors import SdmxQueryError, SdmxTransportError
from dc_sdmx.results import SdmxDataResult, SdmxObservation
from dfo_platform.roles import ROLE_AXES
from graph_release import shadow
from graph_release.coverage import HORIZON, coverage_rows, role_map
from graph_release.dfo import build_dfo_parts, stage_dfo_parts
from graph_release.format import StagedPart, conform_part
from graph_release.gate import (
    Dependency,
    GateResult,
    check_dfo_rebuild,
    check_invariants,
    check_mixer,
    check_sdmx,
    merge,
    read_back,
    refused,
    run_gate,
)
from graph_release.platform import PlatformParts, PlatformRefusal, stage_platform_parts
from graph_release.shadow import (
    PINNED_ARC_PROPERTY,
    PINNED_INDEX_VARIABLES,
    ComparatorBuild,
    diff_coverage,
    diff_origin_arcs,
    index_report,
    shadow_builds,
)
from origin_catalogue.artifact import encode_build, publish_build
from series_coverage.io import (
    write_coverage_table,
    write_current,
    write_facet_metadata,
    write_manifest,
    write_role_names,
)
from series_coverage.schema import CoverageManifest, CoveragePointer, FacetMetadata
from tests.graph_release.recorded import (
    SLICE_BASE_PARTIES,
    SLICE_DECLARATIONS,
    platform_parts,
    slice_platform_parts,
)
from tests.origin_catalogue.support import (
    BUILT_AT,
    build_arc_tables,
    held_wire_arcs,
    sample_arc_tables,
    sample_tables,
)
from tests.series_coverage.support import build_dir, build_test_artifact, pointer_path

_IDS_FIXTURE = Path(__file__).parents[1] / "fixtures/dc_graph/visible_arcs/ids_variables.json"
_STAMP = "stamp-a"
_SEED = "seed-a"


@pytest.fixture
def parts() -> PlatformParts:
    return platform_parts()


def _replace(parts: PlatformParts, **tables: pa.Table) -> PlatformParts:
    """`parts` with each table in `tables` conformed to its part."""
    return dataclasses.replace(
        parts, **{name: conform_part(table, name=name) for name, table in tables.items()}
    )


def _codes(result: GateResult | Sequence) -> list[str]:
    findings = result.findings if isinstance(result, GateResult) else result
    return [str(finding.code) for finding in findings]


# ---------------------------------------------------------------- invariants


def _first_variable(parts: PlatformParts, *, roles: int) -> str:
    """A live variable declaring `roles` roles."""
    declared = role_map(parts.edges, ROLE_AXES)
    for variable, names in declared.items():
        if sum(name is not None for name in (names.donor, names.recipient)) == roles:
            return variable
    raise AssertionError(f"the fixture holds no variable with {roles} role(s)")


def _plus_one_observation(parts: PlatformParts) -> PlatformParts:
    series = parts.series
    counts = pc.add(series["observation_count"], pa.scalar(1, pa.uint32()))
    index = series.schema.get_field_index("observation_count")
    return _replace(parts, series=series.set_column(index, "observation_count", counts))


def _without_first_facet(parts: PlatformParts) -> PlatformParts:
    return _replace(parts, facets=parts.facets.slice(1))


def _untyped(parts: PlatformParts) -> PlatformParts:
    """A coverage variable whose node is no longer a `StatisticalVariable`, so it has no arcs
    of a variable."""
    variable = _first_variable(parts, roles=2)
    rows = parts.nodes.to_pylist()
    for row in rows:
        if row["subject_id"] == variable:
            row["types"] = ["Thing"]
    return _replace(parts, nodes=pa.Table.from_pylist(rows))


def _with_edge(parts: PlatformParts, subject: str, predicate: str, target: str) -> PlatformParts:
    edges = parts.edges.cast(pa.schema([(name, pa.string()) for name in parts.edges.column_names]))
    added = pa.table(
        {
            "subject_id": [subject],
            "predicate": [predicate],
            "object_id": [target],
            "provenance": ["provenance/OECDDAC1"],
        }
    )
    return _replace(parts, edges=pa.concat_tables([edges, added]))


def _dangling(parts: PlatformParts) -> PlatformParts:
    return _with_edge(parts, _first_variable(parts, roles=1), "measuredProperty", "noSuchNode")


def _undeclared_role(parts: PlatformParts) -> PlatformParts:
    """A variable declaring a second role, a property `ROLE_AXES` has no axis for."""
    properties = [
        row["subject_id"]
        for row in parts.nodes.to_pylist()
        if "Property" in (row["types"] or ()) and row["subject_id"] not in ROLE_AXES
    ]
    variable = _first_variable(parts, roles=1)
    return _with_edge(parts, variable, "observationProperties", properties[0])


@pytest.mark.parametrize(
    ("break_parts", "code"),
    [
        (_plus_one_observation, "popcount_mismatch"),
        (_without_first_facet, "unknown_facet"),
        (_untyped, "subset_chain"),
        (_undeclared_role, "missing_role_declaration"),
        (_dangling, "dangling_edge"),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_each_invariant_fails_on_a_release_built_to_break_it(
    parts: PlatformParts, break_parts: Callable[[PlatformParts], PlatformParts], code: str
) -> None:
    assert check_invariants(parts, role_axes=ROLE_AXES) == ()

    assert _codes(check_invariants(break_parts(parts), role_axes=ROLE_AXES)) == [code]


def test_a_platform_refusal_is_a_finding_under_its_own_code() -> None:
    result = refused(PlatformRefusal("unsupported_observation_period", "P1M"))

    assert _codes(result) == ["unsupported_observation_period"]
    assert not result.passed


def test_merge_keeps_every_finding_and_refuses_a_report_twice() -> None:
    first = refused(PlatformRefusal("dangling_edge", "x"))
    second = GateResult(reports={"mixer": {}})

    assert _codes(merge(first, second)) == ["dangling_edge"]
    with pytest.raises(ValueError, match="mixer"):
        merge(second, second)


# ---------------------------------------------------------------- mixer


@pytest.fixture(scope="module")
def ids() -> dict:
    return json.loads(_IDS_FIXTURE.read_text())


def _ids_parts(ids: dict) -> PlatformParts:
    """Platform parts holding the five IDS variables' recorded edges, a `Node` row for each
    variable and each described target, and one series row per variable."""
    rows = ids["spanner_rows"]
    nodes: dict[str, dict] = {
        variable: {
            "subject_id": variable,
            "name": variable,
            "types": ["StatisticalVariable"],
            "value": None,
            "authored": False,
        }
        for variable in ids["variables"]
    }
    for row in rows:
        if row["value"] is not None or row["name"] is not None or row["types"]:
            nodes.setdefault(
                row["object_id"],
                {
                    "subject_id": row["object_id"],
                    "name": row["name"],
                    "types": row["types"],
                    "value": row["value"],
                    "authored": False,
                },
            )
    edges = pa.Table.from_pylist(
        [
            {name: row[name] for name in ("subject_id", "predicate", "object_id", "provenance")}
            for row in rows
        ]
    )
    series = pa.Table.from_pylist(
        [
            {
                "variable_measured": variable,
                "entity1": "country/KEN",
                "extra_entities_id": "",
                "facet_id": "1",
                "entity2": None,
                "observation_count": 0,
                "earliest_date": None,
                "latest_date": None,
                "period_mask_0": 0,
                "period_mask_1": 0,
            }
            for variable in ids["variables"]
        ]
    )
    empty = pa.table({"facet_id": pa.array([], pa.string())})
    return PlatformParts(
        series=series,
        facets=empty,
        edges=conform_part(edges, name="edges"),
        nodes=conform_part(pa.Table.from_pylist(list(nodes.values())), name="nodes"),
        imports=empty,
    )


def _recorded_mixer(ids: dict, monkeypatch: pytest.MonkeyPatch) -> GraphClient:
    """A `GraphClient` whose every `/v2/node` POST answers with the recorded response."""

    def post(self, path, payload, *, deadline=None):
        return ids["mixer_response"]

    monkeypatch.setattr(GraphClient, "post", post)
    return GraphClient(base_url="http://fake/core/api")


class _Recording:
    """An `ArcReader` that records the dcids it is asked for and holds each with no arcs."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def node_arcs_many(self, dcids: Sequence[str], *, properties: Sequence[str]):
        self.asked.extend(dcids)
        return {dcid: {} for dcid in dcids}

    def node_arcs_batch(self, dcids: list[str]):
        return {}


def test_the_mixer_differential_passes_on_the_recorded_mixer_answer(ids, monkeypatch):
    result = check_mixer(
        _ids_parts(ids),
        previous_edges=None,
        graph=_recorded_mixer(ids, monkeypatch),
        seed=_SEED,
    )

    assert result.passed, result.summary()
    assert result.reports["mixer"] == {"changed": 5, "sampled": 5}
    assert result.reports["mixer_only"] == {}


def test_an_arc_the_release_lacks_is_a_mixer_mismatch(ids, monkeypatch):
    parts = _ids_parts(ids)
    variable = ids["variables"][0]
    edges = parts.edges.filter(
        pc.invert(
            pc.and_(
                pc.equal(parts.edges["subject_id"].cast(pa.string()), variable),
                pc.equal(parts.edges["predicate"].cast(pa.string()), "statType"),
            )
        )
    )
    graph = _recorded_mixer(ids, monkeypatch)
    assert check_mixer(parts, previous_edges=None, graph=graph, seed=_SEED).passed

    result = check_mixer(
        dataclasses.replace(parts, edges=edges), previous_edges=None, graph=graph, seed=_SEED
    )

    assert _codes(result) == ["mixer_arc_mismatch"]
    assert result.findings[0].examples[0].startswith(f"{variable} statType")


def test_a_variable_the_mixer_omits_is_a_mismatch_and_a_read_error_is_unavailable(ids, monkeypatch):
    parts = _ids_parts(ids)
    graph = _recorded_mixer(ids, monkeypatch)
    omitted = ids["variables"][1]

    class Omitting:
        def __init__(self, times: int) -> None:
            self.times = times

        def node_arcs_many(self, dcids, *, properties):
            served = graph.node_arcs_many(dcids, properties=properties)
            if omitted in dcids and self.times:
                self.times -= 1
                del served[omitted]
            return served

        def node_arcs_batch(self, dcids):
            return graph.node_arcs_batch(dcids)

    class Failing:
        def node_arcs_many(self, dcids, *, properties):
            raise GraphReadError("503 from /v2/node", upstream_status=503)

    assert check_mixer(parts, previous_edges=None, graph=graph, seed=_SEED).passed
    once = check_mixer(parts, previous_edges=None, graph=Omitting(1), seed=_SEED)
    missing = check_mixer(parts, previous_edges=None, graph=Omitting(2), seed=_SEED)
    failed = check_mixer(parts, previous_edges=None, graph=Failing(), seed=_SEED)

    assert once.passed, once.summary()
    assert _codes(missing) == ["mixer_arc_mismatch"]
    assert missing.findings[0].examples == (f"{omitted}: the mixer holds no node",)
    assert failed.findings == ()
    assert [u.dependency for u in failed.unavailable] == [Dependency.MIXER]
    assert not failed.passed


def test_a_property_only_the_mixer_holds_is_reported_and_decides_nothing(ids, monkeypatch):
    """The release here holds no `constraintProperties`, so the report sees the mixer's; an
    import-added target is hidden, as every reader hides it."""
    graph = _recorded_mixer(ids, monkeypatch)
    parts = _ids_parts(ids)
    parts = dataclasses.replace(
        parts,
        edges=parts.edges.filter(
            pc.not_equal(parts.edges["predicate"].cast(pa.string()), PINNED_ARC_PROPERTY)
        ),
    )
    variable = ids["variables"][3]
    import_added = sorted(IMPORT_ADDED_CONSTRAINT_PROPERTIES)[0]

    class Extra:
        def node_arcs_many(self, dcids, *, properties):
            return graph.node_arcs_many(dcids, properties=properties)

        def node_arcs_batch(self, dcids):
            served = graph.node_arcs_batch(dcids)
            for dcid in dcids:
                served[dcid] = {
                    **served[dcid],
                    PINNED_ARC_PROPERTY: {"nodes": [{"dcid": import_added}]},
                }
            served[variable]["extraProperty"] = {"nodes": [{"value": "x"}]}
            return served

    result = check_mixer(parts, previous_edges=None, graph=Extra(), seed=_SEED)

    assert result.passed, result.summary()
    assert result.reports["mixer_only"] == {"extraProperty": 1}


def test_the_mixer_sample_takes_changed_variables_first_then_seeded_others(ids, monkeypatch):
    parts = _ids_parts(ids)
    graph = _recorded_mixer(ids, monkeypatch)
    variable = ids["variables"][2]
    before = parts.edges.filter(pc.not_equal(parts.edges["subject_id"].cast(pa.string()), variable))

    unchanged = check_mixer(parts, previous_edges=parts.edges, graph=graph, seed=_SEED)
    one = check_mixer(parts, previous_edges=before, graph=graph, seed=_SEED)

    assert unchanged.reports["mixer"] == {"changed": 0, "sampled": 5}
    assert one.reports["mixer"] == {"changed": 1, "sampled": 5}

    monkeypatch.setattr("graph_release.gate.MIXER_CHANGED_CAP", 1)
    monkeypatch.setattr("graph_release.gate.MIXER_RANDOM", 1)
    first, second = _Recording(), _Recording()
    check_mixer(parts, previous_edges=before, graph=first, seed=_SEED)
    check_mixer(parts, previous_edges=before, graph=second, seed=_SEED)
    assert variable in first.asked
    assert len(first.asked) == 2
    assert first.asked == second.asked


# ---------------------------------------------------------------- SDMX


class _Sdmx:
    """A `SeriesReader` answering each request with `count(variable, entities, facet)` rows."""

    def __init__(self, count: Callable[[str, Mapping[str, Sequence[str]], str], int]) -> None:
        self.count = count
        self.requests: list[tuple[str, dict, str]] = []

    def data(self, *, variables, entities, facet_ids=None) -> SdmxDataResult:
        (variable,), (facet,) = variables, facet_ids
        self.requests.append((variable, dict(entities), facet))
        row = SdmxObservation(
            variable=variable,
            role_values={},
            unit=None,
            measurement_method=None,
            observation_period="P1Y",
            provenance=None,
            time_period="2020",
            value="1",
            scaling_factor=None,
            facet_id=facet,
        )
        n = self.count(variable, entities, facet)
        return SdmxDataResult(roles=tuple(entities), rows=(row,) * n)


def _series_counts(parts: PlatformParts) -> Callable[..., int]:
    """What `/data` returns for each series of `parts`, keyed on the entity cells in order."""
    counts = {}
    for row in parts.series.to_pylist():
        entities = tuple(e for e in (row["entity1"], row["entity2"], row.get("entity3")) if e)
        counts[(row["variable_measured"], entities, row["facet_id"])] = row["observation_count"]

    def count(variable, entities, facet):
        return counts[(variable, tuple(v for (v,) in entities.values()), facet)]

    return count


def test_sdmx_counts_match_the_series_rows_they_sample(parts):
    sdmx, again = _Sdmx(_series_counts(parts)), _Sdmx(_series_counts(parts))

    result = check_sdmx(parts, sdmx=sdmx, seed=_SEED)
    check_sdmx(parts, sdmx=again, seed=_SEED)

    assert result.passed, result.summary()
    assert len(sdmx.requests) == 30
    assert again.requests == sdmx.requests


def test_sdmx_requests_fill_the_sorted_roles_with_the_series_entities(parts):
    variable = _first_variable(parts, roles=2)
    declared = role_map(parts.edges, ROLE_AXES)[variable]
    keep = pc.equal(parts.series["variable_measured"].cast(pa.string()), variable)
    only = _replace(parts, series=parts.series.filter(keep))
    sdmx = _Sdmx(_series_counts(only))

    assert check_sdmx(only, sdmx=sdmx, seed=_SEED).passed

    rows = {(r["entity1"], r["entity2"], r["facet_id"]) for r in only.series.to_pylist()}
    assert sdmx.requests
    for requested, entities, facet in sdmx.requests:
        assert requested == variable
        assert list(entities) == [declared.donor, declared.recipient]
        assert (entities[declared.donor][0], entities[declared.recipient][0], facet) in rows


def test_an_sdmx_count_short_by_one_is_a_finding(parts):
    counts = _series_counts(parts)
    first = []

    def short(variable, entities, facet):
        first.append(variable)
        return counts(variable, entities, facet) - (len(first) == 1)

    result = check_sdmx(parts, sdmx=_Sdmx(short), seed=_SEED)

    assert _codes(result) == ["sdmx_count_mismatch"]
    assert result.findings[0].detail.startswith("1 of 30")


def test_a_refused_sdmx_query_is_a_finding_and_a_transport_error_unavailable(parts):
    def refuse(*_):
        raise SdmxQueryError("reserved character")

    def drop(*_):
        raise SdmxTransportError("connection reset")

    assert _codes(check_sdmx(parts, sdmx=_Sdmx(refuse), seed=_SEED)) == ["sdmx_count_mismatch"]
    unavailable = check_sdmx(parts, sdmx=_Sdmx(drop), seed=_SEED)
    assert unavailable.findings == ()
    assert [u.dependency for u in unavailable.unavailable] == [Dependency.SDMX]


# ---------------------------------------------------------------- shadow


def _publish_coverage(base: Path, parts: PlatformParts, *, stamp: str = _STAMP) -> str:
    """Publishes the coverage build `parts` give, as the coverage job writes one."""
    roles = role_map(parts.edges, ROLE_AXES)
    table = coverage_rows(parts.series, roles)
    build_id = f"{stamp}-build"
    prefix = build_dir(base, build_id)
    prefix.mkdir(parents=True)
    write_coverage_table(table.cast(HORIZON.schema), prefix / "coverage.parquet", horizon=HORIZON)
    write_role_names(dict(roles), prefix / "role_names.json")
    facets = {
        facet: FacetMetadata()
        for facet in parts.series["facet_id"].cast(pa.string()).unique().to_pylist()
    }
    write_facet_metadata(facets, prefix / "facets.json")
    write_manifest(
        CoverageManifest(
            stamp=stamp,
            build_id=build_id,
            built_at="2026-10-19T00:00:00Z",
            source="tests.graph_release.test_gate",
            row_counts={"rows": table.num_rows},
            files={
                "coverage": "coverage.parquet",
                "role_names": "role_names.json",
                "facets": "facets.json",
            },
            horizon=HORIZON,
        ),
        prefix / "manifest.json",
    )
    write_current(CoveragePointer(stamp=stamp, build_id=build_id), pointer_path(base))
    return f"file://{base}"


def _rewrite_published(base: Path, change: Callable[[pa.Table], pa.Table]) -> None:
    path = build_dir(base, f"{_STAMP}-build") / "coverage.parquet"
    table = pq.read_table(path)
    write_coverage_table(change(table).cast(HORIZON.schema), path, horizon=HORIZON)


def _bump_first_count(table: pa.Table) -> pa.Table:
    counts = table["n_obs"].to_pylist()
    counts[0] += 1
    return table.set_column(
        table.schema.get_field_index("n_obs"), "n_obs", pa.array(counts, pa.uint32())
    )


def test_the_coverage_diff_finds_a_changed_or_missing_row(parts, tmp_path):
    base = tmp_path / "coverage"
    uri = _publish_coverage(base, parts)
    clean = diff_coverage(parts, import_stamp=_STAMP, role_axes=ROLE_AXES, uri=uri)
    assert clean.passed, clean.summary()

    _rewrite_published(base, _bump_first_count)
    changed = diff_coverage(parts, import_stamp=_STAMP, role_axes=ROLE_AXES, uri=uri)
    _rewrite_published(base, lambda table: table.slice(1))
    missing = diff_coverage(parts, import_stamp=_STAMP, role_axes=ROLE_AXES, uri=uri)
    _rewrite_published(base, _with_unknown_recipient)  # and still without the first row
    extra = diff_coverage(parts, import_stamp=_STAMP, role_axes=ROLE_AXES, uri=uri)

    assert _codes(changed) == ["shadow_coverage_diff"]
    assert changed.findings[0].detail.startswith("1 coverage row")
    assert _codes(missing) == ["shadow_coverage_diff"]
    assert missing.findings[0].detail.startswith("1 coverage row")
    assert _codes(extra) == ["shadow_coverage_diff"]
    assert extra.findings[0].detail.startswith("2 coverage row")
    assert "country/ZZZ" in [example.split()[-1] for example in extra.findings[0].examples]


def _with_unknown_recipient(table: pa.Table) -> pa.Table:
    """`table` plus a copy of its first row for a recipient no release row names."""
    row = table.slice(0, 1).to_pylist()[0] | {"recipient": "country/ZZZ"}
    return pa.concat_tables([table, pa.Table.from_pylist([row], schema=table.schema)])


def test_the_coverage_diff_finds_other_role_names(parts, tmp_path):
    base = tmp_path / "coverage"
    uri = _publish_coverage(base, parts)
    path = build_dir(base, f"{_STAMP}-build") / "role_names.json"
    names = json.loads(path.read_text())
    variable = next(v for v, n in names.items() if n["donor"] and n["recipient"])
    names[variable] = {"donor": names[variable]["recipient"], "recipient": names[variable]["donor"]}
    path.write_text(json.dumps(names))

    result = diff_coverage(parts, import_stamp=_STAMP, role_axes=ROLE_AXES, uri=uri)

    assert _codes(result) == ["shadow_coverage_diff"]
    assert result.findings[0].examples == (variable,)


def test_a_comparator_at_another_import_or_unreadable_is_unavailable(parts, tmp_path):
    uri = _publish_coverage(tmp_path / "coverage", parts, stamp="stamp-old")

    behind = diff_coverage(parts, import_stamp=_STAMP, role_axes=ROLE_AXES, uri=uri)
    absent = diff_coverage(
        parts, import_stamp=_STAMP, role_axes=ROLE_AXES, uri=f"file://{tmp_path / 'none'}"
    )
    no_origin = diff_origin_arcs(parts, import_stamp=_STAMP, uri=f"file://{tmp_path / 'none'}")

    for result, dependency in (
        (behind, Dependency.COVERAGE_BUILD),
        (absent, Dependency.COVERAGE_BUILD),
        (no_origin, Dependency.ORIGIN_CATALOGUE),
    ):
        assert result.findings == ()
        assert [u.dependency for u in result.unavailable] == [dependency]


def _mixer_answers(ids: dict) -> dict[str, dict]:
    """The recorded `->*` answer by variable, a copy each test may edit."""
    return json.loads(json.dumps({v: n["arcs"] for v, n in ids["mixer_response"]["data"].items()}))


def _publish_origin(base: Path, answers: Mapping[str, dict], *, stamp: str = _STAMP) -> str:
    """Publishes the sample origin build with `answers`' variables as its DFO variables and
    their `->*` answers as its DFO arc table rows, as the catalogue job stores them."""
    uri = f"file://{base}"
    sample = sample_arc_tables()
    arcs = build_arc_tables(
        dfo=answers,
        base={node: held_wire_arcs(sample, node) or None for node in sample.base},
        schema={node: held_wire_arcs(sample, node) or {} for node in sample.schema},
    )
    build = encode_build(
        dataclasses.replace(sample_tables(), dfo_variables=frozenset(answers)),
        build_id="build-a",
        built_at=BUILT_AT,
        build_seconds=1.0,
        identity_inputs={"dfo_fingerprint": stamp, "registry": "r1", "week": "2026-W42"},
        registry_hash="r1",
        arcs=arcs,
    )
    assert publish_build(uri, build, expected_generation=None)
    return uri


def test_the_origin_arc_diff_pins_the_import_added_property_and_finds_any_other(ids, tmp_path):
    parts = _ids_parts(ids)
    lacking, renamed = _mixer_answers(ids), _mixer_answers(ids)
    for arcs in lacking.values():
        del arcs["statType"]
    for arcs in renamed.values():
        for node in arcs["constraintProperties"]["nodes"]:
            if node["dcid"] == "sourceIndicatorCode":
                node["dcid"] = "sourceSeriesCode"

    clean = diff_origin_arcs(
        parts, import_stamp=_STAMP, uri=_publish_origin(tmp_path / "a", _mixer_answers(ids))
    )
    assert clean.passed, clean.summary()
    assert clean.reports["shadow_arcs"] == {"variables": 5, "pinned": 5}

    for answers, name, side in (
        (lacking, "statType", "release only"),
        (renamed, "sourceSeriesCode", "origin catalogue only"),
    ):
        result = diff_origin_arcs(
            parts, import_stamp=_STAMP, uri=_publish_origin(tmp_path / name, answers)
        )
        assert _codes(result) == ["shadow_arc_diff"], name
        assert result.findings[0].detail.startswith("5 arc(s)"), name
        assert all(name in e and e.endswith(side) for e in result.findings[0].examples), name


def test_shadow_builds_names_each_comparator_build_and_its_import(parts, ids, tmp_path):
    coverage = build_test_artifact(tmp_path / "coverage", stamp=_STAMP)
    origin = _publish_origin(tmp_path / "origin", _mixer_answers(ids))
    old_origin = _publish_origin(tmp_path / "old", _mixer_answers(ids), stamp="stamp-old")

    assert shadow_builds(coverage_uri=coverage, origin_uri=origin) == (
        ComparatorBuild(Dependency.COVERAGE_BUILD, _STAMP, f"{_STAMP}-build"),
        ComparatorBuild(Dependency.ORIGIN_CATALOGUE, _STAMP, "build-a"),
    )
    assert shadow_builds(coverage_uri=coverage, origin_uri=old_origin) == (
        ComparatorBuild(Dependency.COVERAGE_BUILD, _STAMP, f"{_STAMP}-build"),
        ComparatorBuild(Dependency.ORIGIN_CATALOGUE, "stamp-old", "build-a"),
    )

    missing = shadow_builds(coverage_uri=coverage, origin_uri=f"file://{tmp_path / 'x'}")
    assert getattr(missing, "dependency", None) == Dependency.ORIGIN_CATALOGUE
    pointer_path(tmp_path / "coverage").write_text("{")
    malformed = shadow_builds(coverage_uri=coverage, origin_uri=origin)
    assert getattr(malformed, "dependency", None) == Dependency.COVERAGE_BUILD


@pytest.mark.parametrize("failing", [Dependency.COVERAGE_BUILD, Dependency.ORIGIN_CATALOGUE])
def test_shadow_builds_count_a_credentials_failure_as_unavailable(
    ids, tmp_path, monkeypatch, failing
):
    coverage = build_test_artifact(tmp_path / "coverage", stamp=_STAMP)
    origin = _publish_origin(tmp_path / "origin", _mixer_answers(ids))
    broken = coverage if failing is Dependency.COVERAGE_BUILD else origin
    read = shadow.read_uri_bytes

    def refreshing(uri, *args, **kwargs):
        if uri.startswith(broken):
            raise RefreshError("token refresh failed")
        return read(uri, *args, **kwargs)

    monkeypatch.setattr(shadow, "read_uri_bytes", refreshing)

    result = shadow_builds(coverage_uri=coverage, origin_uri=origin)

    assert getattr(result, "dependency", None) == failing


def test_the_index_report_marks_the_pinned_variable_and_counts_the_rest(tmp_path):
    pinned = next(iter(PINNED_INDEX_VARIABLES))
    committed = {
        "statvars": [{"dcid": "A", "name": "a", "provenances": ["p"]}],
        "listed_statvars": [{"dcid": "A"}],
    }
    release = {
        "statvars": [{"dcid": "A", "name": "a2", "provenances": ["q"]}],
        "listed_statvars": [{"dcid": "A"}, {"dcid": pinned}, {"dcid": "B"}],
    }
    path = tmp_path / "index.json.gz"
    path.write_bytes(gzip.compress(json.dumps(committed).encode()))

    report = index_report(gzip.compress(json.dumps(release).encode()), path)

    assert report["statvars"]["changed_fields"] == {"name": 1}
    assert report["statvars"]["unpinned"] == 1
    assert report["listed_statvars"]["added_count"] == 2
    assert (report["listed_statvars"]["pinned"], report["listed_statvars"]["unpinned"]) == (1, 1)


# ---------------------------------------------------------------- DFO rebuild and run_gate


def _staged_release(tmp_path: Path) -> tuple[PlatformParts, list[StagedPart]]:
    """The recorded slice's platform and DFO parts, staged as the builder stages them."""
    parts = slice_platform_parts()
    staged = [
        *stage_platform_parts(parts, staging_dir=tmp_path, execution_id="exec-1"),
        *stage_dfo_parts(
            build_dfo_parts(parts, SLICE_DECLARATIONS, SLICE_BASE_PARTIES),
            staging_dir=tmp_path,
            execution_id="exec-1",
        ),
    ]
    return parts, staged


def _drop_crs_variable(staged: list[StagedPart]) -> str:
    """Rewrites one CRS `series` file in place without one variable's rows, keeping its name.
    Returns the variable. The file is read into memory: a mapped read leaves the filtered
    table's dictionaries in the mapping, which truncating the file zeroes on Linux."""
    part = next(
        part
        for part in staged
        if part.ref.name == "series"
        and part.ref.provenance == SLICE_DECLARATIONS.datasets[0].provenance
    )
    table = pa.ipc.open_file(pa.py_buffer(part.path.read_bytes())).read_all()
    variable = table["variable_measured"].cast(pa.string())[0].as_py()
    kept = table.filter(pc.not_equal(table["variable_measured"].cast(pa.string()), variable))
    with pa.OSFile(str(part.path), "wb") as sink, pa.ipc.new_file(sink, kept.schema) as writer:
        writer.write_table(kept)
    return variable


def test_a_corrupted_platform_file_fails_the_dfo_rebuild(tmp_path):
    parts, staged = _staged_release(tmp_path)
    assert (
        check_dfo_rebuild(read_back(staged), staged=staged, declarations=SLICE_DECLARATIONS) == ()
    )

    _drop_crs_variable(staged)

    assert check_dfo_rebuild(parts, staged=staged, declarations=SLICE_DECLARATIONS) == ()
    result = check_dfo_rebuild(read_back(staged), staged=staged, declarations=SLICE_DECLARATIONS)
    assert _codes(result) == ["dfo_rebuild_mismatch"]
    assert result[0].examples == ("code_index_blocks",)


def test_a_dfo_rebuild_the_export_refuses_names_the_refusal(tmp_path):
    parts, staged = _staged_release(tmp_path)
    base_parties = next(part for part in staged if part.ref.name == "base_parties")
    base_parties.path.write_bytes(b"{}")

    result = check_dfo_rebuild(parts, staged=staged, declarations=SLICE_DECLARATIONS)
    assert _codes(result) == ["dfo_rebuild_mismatch"]
    assert result[0].examples == ("code_index_export_refused",)


def test_run_gate_reads_the_staged_files_and_never_passes_an_unavailable_dependency(tmp_path):
    _, staged = _staged_release(tmp_path)

    class Failing:
        def node_arcs_many(self, dcids, *, properties):
            raise GraphReadError("connection reset")

    def drop(*_):
        raise SdmxTransportError("connection reset")

    result = run_gate(
        staged,
        declarations=SLICE_DECLARATIONS,
        previous=None,
        graph=Failing(),
        sdmx=_Sdmx(drop),
        seed=_SEED,
    )

    assert result.findings == ()
    assert [u.dependency for u in result.unavailable] == [Dependency.MIXER, Dependency.SDMX]
    assert not result.passed
    assert result.reports["rows"]["series"] == platform_parts().series.num_rows

    _drop_crs_variable(staged)
    broken = run_gate(
        staged,
        declarations=SLICE_DECLARATIONS,
        previous=None,
        graph=Failing(),
        sdmx=_Sdmx(drop),
        seed=_SEED,
    )
    assert "dfo_rebuild_mismatch" in _codes(broken)
