"""Checks report tables, ceiling bands, thinking levels, and offline rescoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptlab import report
from promptlab.graphcache import GraphCache
from promptlab.report import MetricTable, ReportError
from promptlab.store import CellManifest, CellStore
from search.engine.bind import _build_universal_bind_prompt, _UniversalBindOutput
from search.engine.shape import ShapeDraft
from search.eval.cases import CHECK_NAMES, Case, validate_case
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _cell_manifest(cell_id: str, *, rung_name: str, capture_id: str = "cap-1") -> CellManifest:
    return CellManifest(
        cell_id=cell_id,
        rung={"name": rung_name},
        capture_id=capture_id,
        git_sha="abc123",
        model="test-model",
        seeds=(1, 2),
        extract_system="",
        bind_system="",
    )


def _bind_record(
    *,
    case_id: str,
    leg: int,
    seed: int,
    prompt: str,
    output: dict,
    scores: dict,
    thinking_level: str = "minimal",
) -> dict:
    return {
        "case_id": case_id,
        "leg": leg,
        "seed": seed,
        "stage": "bind",
        "schema": "_UniversalBindOutput",
        "system_sha1": "s" * 40,
        "prompt_sha1": "p" * 40,
        "prompt": prompt,
        "output": output,
        "usage": None,
        "elapsed_ms": 1,
        "llm_seed": None,
        "thinking_level": thinking_level,
        "scores": scores,
    }


def _e2e_record(
    *,
    case_id: str,
    leg: int,
    seed: int,
    output: dict,
    checks: dict[str, dict],
    bind_thinking_level: str = "minimal",
) -> dict:
    return {
        "case_id": case_id,
        "leg": leg,
        "seed": seed,
        "output": output,
        "checks": checks,
        "llm_calls": [
            {"stage": "extract", "thinking_level": None},
            {"stage": "bind", "thinking_level": bind_thinking_level},
        ],
    }


def _checks(**verdicts: str) -> dict[str, dict]:
    return {name: {"verdict": verdict, "detail": ""} for name, verdict in verdicts.items()}


def _make_store(tmp_path: Path, cell_id: str, rung_name: str, records: list[dict]) -> CellStore:
    """Write a complete cell and reopen it for report operations.

    A freshly opened store starts with `_finished` False, as in CLI use. Reusing the finished
    instance would make `rescore_cell` raise "already finished".
    """
    store = CellStore.create(
        tmp_path / cell_id, manifest=_cell_manifest(cell_id, rung_name=rung_name)
    )
    for record in records:
        record = dict(record)
        case_id = record.pop("case_id")
        leg = record.pop("leg")
        seed = record.pop("seed")
        store.append(case_id=case_id, leg=leg, seed=seed, call=record)
    store.finish({})
    return CellStore.open(store.root)


def _candidates_case(case_id: str, *, stat_vars: list[str]) -> Case:
    candidates: dict = (
        {"measure": {"shape": {"population_type": "Person"}, "stat_vars": stat_vars}}
        if stat_vars
        else {}
    )
    return validate_case(
        {
            "id": case_id,
            "query": "a query",
            "why": "A candidates case for rescoring.",
            "expect": {"candidates": candidates},
        },
        families={},
    )


# ---------------------------------------------------------------------------
# open_cell
# ---------------------------------------------------------------------------


def test_open_cell_refuses_a_partial_cell_without_allow_partial(tmp_path: Path):
    store = CellStore.create(
        tmp_path / "cells" / "c1", manifest=_cell_manifest("c1", rung_name="r")
    )
    store.append(
        case_id="g1",
        leg=0,
        seed=1,
        call=_bind_record(case_id="g1", leg=0, seed=1, prompt="p", output={}, scores={}),
    )
    store.close()  # no finish() -- stays partial

    with pytest.raises(ReportError, match="not complete"):
        report.open_cell(tmp_path / "cells", "c1", allow_partial=False)

    opened = report.open_cell(tmp_path / "cells", "c1", allow_partial=True)
    assert opened.manifest.cell_id == "c1"


def test_open_cell_raises_report_error_for_a_missing_cell(tmp_path: Path):
    with pytest.raises(ReportError):
        report.open_cell(tmp_path / "cells", "no-such-cell", allow_partial=False)


# ---------------------------------------------------------------------------
# build_metric_table -- bind/extract-shaped records
# ---------------------------------------------------------------------------


def test_build_metric_table_bind_cell_aggregate_and_per_seed(tmp_path: Path):
    records = [
        _bind_record(
            case_id="g1",
            leg=0,
            seed=1,
            prompt="p",
            output={},
            scores={"shape_election_match": 1.0, "over_nomination_count": 0},
        ),
        _bind_record(
            case_id="g1",
            leg=0,
            seed=2,
            prompt="p",
            output={},
            scores={"shape_election_match": 0.0, "over_nomination_count": 1},
        ),
        _bind_record(
            case_id="g2",
            leg=0,
            seed=1,
            prompt="p",
            output={},
            scores={"shape_election_match": 1.0, "over_nomination_count": 2},
        ),
        _bind_record(
            case_id="g2",
            leg=0,
            seed=2,
            prompt="p",
            output={},
            scores={"shape_election_match": 1.0, "over_nomination_count": 0},
        ),
    ]
    store = _make_store(tmp_path, "bind-cell", "abl-bind-also-plausible", records)

    table = report.build_metric_table(store)

    assert table.rung_name == "abl-bind-also-plausible"
    assert table.n_seeds == 2
    assert table.aggregate["shape_election_match_rate"] == pytest.approx(0.75)
    assert table.aggregate["over_nomination_count_mean"] == pytest.approx(0.75)
    assert table.per_seed[1]["shape_election_match_rate"] == pytest.approx(1.0)
    assert table.per_seed[2]["shape_election_match_rate"] == pytest.approx(0.5)
    assert table.band("shape_election_match_rate") == pytest.approx((0.5, 1.0))
    assert table.band("over_nomination_count_mean") == pytest.approx((0.5, 1.0))
    assert table.band("no-such-metric") is None


# ---------------------------------------------------------------------------
# build_metric_table -- e2e-shaped records
# ---------------------------------------------------------------------------


def test_build_metric_table_e2e_cell_aggregate_and_per_seed(tmp_path: Path):
    records = [
        _e2e_record(
            case_id="g1",
            leg=0,
            seed=1,
            output={"x": 1},
            checks=_checks(measure="pass"),
        ),
        _e2e_record(
            case_id="g1",
            leg=0,
            seed=2,
            output={"x": 1},
            checks=_checks(measure="fail"),
        ),
        _e2e_record(
            case_id="g2",
            leg=0,
            seed=1,
            output={"x": 2},
            checks=_checks(measure="pass"),
        ),
        _e2e_record(
            case_id="g2",
            leg=0,
            seed=2,
            output={"x": 2},
            checks=_checks(measure="pass"),
        ),
    ]
    store = _make_store(tmp_path, "e2e-cell", "production", records)

    table = report.build_metric_table(store)

    assert table.n_seeds == 2
    assert table.aggregate["measure_pass_rate"] == pytest.approx(0.75)
    assert table.per_seed[1]["measure_pass_rate"] == pytest.approx(1.0)
    assert table.per_seed[2]["measure_pass_rate"] == pytest.approx(0.5)
    # Cross-seed: computable only on the full (multi-seed-per-case) set.
    assert "seed_flip_rate" in table.aggregate
    assert "seed_flip_rate" not in table.per_seed[1]
    assert "seed_flip_rate" not in table.per_seed[2]


@pytest.mark.parametrize("name", CHECK_NAMES)
def test_every_check_pass_rate_is_a_higher_is_better_metric(name):
    assert report._metric_direction(f"{name}_pass_rate") == "higher_is_better"


def test_e2e_report_leaves_out_a_check_no_record_scored(tmp_path: Path):
    records = [
        _e2e_record(
            case_id="g1",
            leg=0,
            seed=1,
            output={"x": 1},
            checks=_checks(entities="pass", provenance="n/a", refusal="unverifiable"),
        )
    ]
    store = _make_store(tmp_path, "unscored-cell", "production", records)

    table = report.build_metric_table(store)

    assert table.aggregate["entities_pass_rate"] == 1.0
    assert "provenance_pass_rate" not in table.aggregate
    assert "refusal_pass_rate" not in table.aggregate


def test_build_metric_table_raises_on_a_cell_with_no_records(tmp_path: Path):
    store = _make_store(tmp_path, "empty-cell", "r", [])
    with pytest.raises(ReportError, match="no call records"):
        report.build_metric_table(store)


def test_bind_metrics_surface_alternate_recall_scored_and_unscorable_counts(tmp_path: Path):
    """Render scored and unscorable counts beside the alternate-recall mean."""
    records = [
        _bind_record(
            case_id="g1",
            leg=0,
            seed=1,
            prompt="p",
            output={},
            scores={"expected_alternate_recall": 1.0},
        ),
        _bind_record(
            case_id="g2",
            leg=0,
            seed=1,
            prompt="p",
            output={},
            scores={"expected_alternate_recall": "unscorable"},
        ),
    ]
    store = _make_store(tmp_path, "recall-counts-cell", "abl-bind-x", records)

    table = report.build_metric_table(store)

    assert table.aggregate["expected_alternate_recall_mean"] == pytest.approx(1.0)
    assert table.aggregate["expected_alternate_recall_scored"] == 1
    assert table.aggregate["expected_alternate_recall_unscorable"] == 1

    md = report.render_markdown([table], against=None)
    assert "expected_alternate_recall_scored" in md
    assert "expected_alternate_recall_unscorable" in md

    data = report.render_json([table], against=None)
    metrics = data["cells"][0]["metrics"]
    assert metrics["expected_alternate_recall_scored"]["value"] == 1
    assert metrics["expected_alternate_recall_unscorable"]["value"] == 1


def test_bind_metrics_omit_alternate_recall_counts_when_no_record_carries_the_key(
    tmp_path: Path,
):
    """Cells without recall scores omit the scored and unscorable counts."""
    records = [
        _bind_record(
            case_id="g1",
            leg=0,
            seed=1,
            prompt="p",
            output={},
            scores={"shape_election_match": 1.0},
        ),
    ]
    store = _make_store(tmp_path, "no-recall-cell", "abl-bind-x", records)

    table = report.build_metric_table(store)

    assert table.aggregate["shape_election_match_rate"] == pytest.approx(1.0)
    assert "expected_alternate_recall_scored" not in table.aggregate
    assert "expected_alternate_recall_unscorable" not in table.aggregate


# ---------------------------------------------------------------------------
# require_minimal_ceiling
# ---------------------------------------------------------------------------


def test_require_minimal_ceiling_passes_when_every_bind_call_is_minimal(tmp_path: Path):
    records = [
        _bind_record(case_id="g1", leg=0, seed=1, prompt="p", output={}, scores={}),
        _bind_record(case_id="g2", leg=0, seed=1, prompt="p", output={}, scores={}),
    ]
    store = _make_store(tmp_path, "ceiling-bind", "production", records)
    report.require_minimal_ceiling(store)  # does not raise


def test_require_minimal_ceiling_raises_for_a_non_minimal_bind_call(tmp_path: Path):
    records = [
        _bind_record(case_id="g1", leg=0, seed=1, prompt="p", output={}, scores={}),
        _bind_record(
            case_id="g2", leg=0, seed=1, prompt="p", output={}, scores={}, thinking_level="low"
        ),
    ]
    store = _make_store(tmp_path, "ceiling-bind-bad", "production", records)
    with pytest.raises(ReportError, match="not all thinking_level='minimal'"):
        report.require_minimal_ceiling(store)


def test_require_minimal_ceiling_checks_e2e_llm_calls_by_stage(tmp_path: Path):
    records = [
        _e2e_record(case_id="g1", leg=0, seed=1, output={}, checks={}, bind_thinking_level="low"),
    ]
    store = _make_store(tmp_path, "ceiling-e2e-bad", "production", records)
    with pytest.raises(ReportError, match="not all thinking_level='minimal'"):
        report.require_minimal_ceiling(store)


def test_require_minimal_ceiling_passes_vacuously_for_extract_only_records(tmp_path: Path):
    records = [
        {
            "case_id": "g1",
            "leg": 0,
            "seed": 1,
            "stage": "extract",
            "schema": "Extraction",
            "system_sha1": "s",
            "prompt_sha1": "p",
            "prompt": "q",
            "output": {},
            "usage": None,
            "elapsed_ms": 1,
            "llm_seed": None,
            "thinking_level": None,
            "scores": {},
        },
    ]
    store = _make_store(tmp_path, "ceiling-extract", "production", records)
    report.require_minimal_ceiling(store)  # no bind-stage calls -- passes


# ---------------------------------------------------------------------------
# render_markdown / render_json
# ---------------------------------------------------------------------------


def _table(cell_id: str, rung_name: str, aggregate: dict, per_seed: dict) -> MetricTable:
    return MetricTable(
        cell_id=cell_id,
        rung_name=rung_name,
        n_seeds=len(per_seed),
        aggregate=aggregate,
        per_seed=per_seed,
    )


def test_render_markdown_without_against_has_a_simple_table():
    table = _table("c1", "r1", {"metric_a": 0.5}, {1: {"metric_a": 0.5}})
    out = report.render_markdown([table], against=None)
    assert "| metric | value |" in out
    assert "ceiling band" not in out
    assert "metric_a" in out
    assert "0.5000" in out


def test_render_markdown_with_against_has_band_delta_and_verdicts():
    # Three classified higher-is-better metrics: shape_election_match_rate
    # falls below the ceiling's own band min, entity_dcid_match_rate falls
    # inside it, date_structure_match_rate clears the band max.
    ceiling = _table(
        "ceiling",
        "production",
        aggregate={},
        per_seed={
            1: {
                "shape_election_match_rate": 0.5,
                "entity_dcid_match_rate": 0.5,
                "date_structure_match_rate": 0.5,
            },
            2: {
                "shape_election_match_rate": 0.9,
                "entity_dcid_match_rate": 0.9,
                "date_structure_match_rate": 0.9,
            },
        },
    )
    cell = _table(
        "c1",
        "abl-bind-x",
        aggregate={
            "shape_election_match_rate": 0.3,
            "entity_dcid_match_rate": 0.6,
            "date_structure_match_rate": 0.95,
        },
        per_seed={1: {}},
    )

    out = report.render_markdown([cell], against=ceiling)

    assert "| metric | value | ceiling band | delta | verdict |" in out
    assert "BELOW" in out  # shape_election_match_rate=0.3 < band min 0.5
    assert "INSIDE-BAND" in out  # entity_dcid_match_rate=0.6 inside [0.5, 0.9]
    assert "PASS" in out  # date_structure_match_rate=0.95 > band max 0.9


def test_render_markdown_lower_is_better_verdicts_flip_polarity():
    """For a lower-is-better metric, values above the band fail and values below pass."""
    ceiling = _table(
        "ceiling",
        "production",
        aggregate={},
        per_seed={1: {"over_nomination_count_mean": 0.5}, 2: {"over_nomination_count_mean": 0.9}},
    )
    over_band = _table(
        "c-over", "abl-bind-x", aggregate={"over_nomination_count_mean": 1.2}, per_seed={1: {}}
    )
    under_band = _table(
        "c-under", "abl-bind-y", aggregate={"over_nomination_count_mean": 0.2}, per_seed={1: {}}
    )

    over_out = report.render_markdown([over_band], against=ceiling)
    under_out = report.render_markdown([under_band], against=ceiling)

    assert "BELOW" in over_out  # 1.2 over-nominates worse than the ceiling's own noise ceiling
    assert "PASS" in under_out  # 0.2 under-nominates better than the ceiling's own noise floor


def test_render_markdown_informational_metric_gets_no_verdict():
    ceiling = _table(
        "ceiling",
        "production",
        aggregate={},
        per_seed={
            1: {"expected_alternate_recall_scored": 20.0},
            2: {"expected_alternate_recall_scored": 20.0},
        },
    )
    cell = _table(
        "c1", "abl-bind-x", aggregate={"expected_alternate_recall_scored": 60.0}, per_seed={1: {}}
    )

    out = report.render_markdown([cell], against=ceiling)

    assert "| expected_alternate_recall_scored | 60.0000 | — | — | — |" in out
    assert "PASS" not in out
    assert "BELOW" not in out
    assert "INSIDE-BAND" not in out


def test_render_markdown_unclassified_metric_raises_report_error():
    ceiling = _table("ceiling", "production", aggregate={}, per_seed={1: {"mystery_metric": 0.5}})
    cell = _table("c1", "abl-bind-x", aggregate={"mystery_metric": 0.3}, per_seed={1: {}})

    with pytest.raises(ReportError, match="mystery_metric"):
        report.render_markdown([cell], against=ceiling)


def test_render_json_reports_ceiling_band_delta_and_verdict():
    ceiling = _table(
        "ceiling",
        "production",
        aggregate={},
        per_seed={
            1: {"shape_election_match_rate": 0.5},
            2: {"shape_election_match_rate": 0.9},
        },
    )
    cell = _table(
        "c1",
        "abl-bind-x",
        aggregate={"shape_election_match_rate": 0.3},
        per_seed={1: {"shape_election_match_rate": 0.3}},
    )

    data = report.render_json([cell], against=ceiling)

    entry = data["cells"][0]
    assert entry["cell_id"] == "c1"
    assert entry["rung_name"] == "abl-bind-x"
    metric = entry["metrics"]["shape_election_match_rate"]
    assert metric["value"] == pytest.approx(0.3)
    assert metric["ceiling_band"] == pytest.approx([0.5, 0.9])
    assert metric["delta"] == pytest.approx(-0.2)
    assert metric["verdict"] == "BELOW"


def test_render_json_lower_is_better_verdict_flips_polarity():
    ceiling = _table(
        "ceiling",
        "production",
        aggregate={},
        per_seed={
            1: {"over_nomination_count_mean": 0.5},
            2: {"over_nomination_count_mean": 0.9},
        },
    )
    cell = _table(
        "c1", "abl-bind-x", aggregate={"over_nomination_count_mean": 1.2}, per_seed={1: {}}
    )

    data = report.render_json([cell], against=ceiling)

    metric = data["cells"][0]["metrics"]["over_nomination_count_mean"]
    assert metric["verdict"] == "BELOW"  # over-nominating 1.2 is worse than the ceiling's band


def test_render_json_informational_metric_has_no_verdict_or_band():
    ceiling = _table(
        "ceiling",
        "production",
        aggregate={},
        per_seed={
            1: {"expected_alternate_recall_scored": 20.0},
            2: {"expected_alternate_recall_scored": 20.0},
        },
    )
    cell = _table(
        "c1", "abl-bind-x", aggregate={"expected_alternate_recall_scored": 60.0}, per_seed={1: {}}
    )

    data = report.render_json([cell], against=ceiling)

    metric = data["cells"][0]["metrics"]["expected_alternate_recall_scored"]
    assert metric == {"value": 60.0}


def test_render_json_omits_band_fields_when_metric_absent_from_ceiling():
    ceiling = _table("ceiling", "production", aggregate={}, per_seed={1: {}})
    cell = _table("c1", "abl-bind-x", aggregate={"only_here": 1.0}, per_seed={1: {}})

    data = report.render_json([cell], against=ceiling)

    metric = data["cells"][0]["metrics"]["only_here"]
    assert metric == {"value": 1.0}


def test_render_json_unclassified_metric_raises_report_error():
    ceiling = _table("ceiling", "production", aggregate={}, per_seed={1: {"mystery_metric": 0.5}})
    cell = _table("c1", "abl-bind-x", aggregate={"mystery_metric": 0.3}, per_seed={1: {}})

    with pytest.raises(ReportError, match="mystery_metric"):
        report.render_json([cell], against=ceiling)


# ---------------------------------------------------------------------------
# --rescore
# ---------------------------------------------------------------------------


_SHAPE_DISTRACTOR = ShapeDraft(
    shape_id="s0",
    label="distractor shape",
    pop_type_dcid="Household",
    meas_prop_dcid="income",
    stat_type_dcid="medianValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(),
    slot_taxonomy={},
    representative_score=1.0,
)


def _hit_shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="s1",
        label="the expected shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        slot_taxonomy={},
        representative_score=1.0,
    )


_HIT_ARCS = {
    "populationType": {"nodes": [{"dcid": "Person"}]},
    "measuredProperty": {"nodes": [{"dcid": "count"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}


def _seed_graph_cache(path: Path, *, cached_dcid: str, arcs: dict) -> None:
    """Write one node-arcs entry to a GraphCache at `path`."""
    graph = GraphCache(
        FakeGraph(nodes={cached_dcid: {"type": "StatisticalVariable", "arcs": arcs}}),
        path=path,
        read_only=False,
    )
    graph.node_arcs(cached_dcid)  # write-back: caches this dcid's node_arcs
    graph.close()


def test_rescore_cell_resolves_a_cached_alternate_and_treats_a_cache_miss_as_unscorable(
    tmp_path: Path,
):
    dcid_hit = "sdg/HIT_SV"
    dcid_uncached = "sdg/UNCACHED_SV"  # never written to the graph store
    dcid_only_uncached = "sdg/ONLY_UNCACHED_SV"

    graph_path = tmp_path / "graph.sqlite"
    _seed_graph_cache(graph_path, cached_dcid=dcid_hit, arcs=_HIT_ARCS)

    prompt = _build_universal_bind_prompt("some variable", [_SHAPE_DISTRACTOR, _hit_shape()])
    output = _UniversalBindOutput(
        chosen_shape_index=1, bindings=[], also_plausible=[1], ask=None
    ).model_dump(mode="json")

    records = [
        _bind_record(
            case_id="g-hit",
            leg=0,
            seed=1,
            prompt=prompt,
            output=output,
            scores={},
        ),
        _bind_record(
            case_id="g-all-unscorable",
            leg=0,
            seed=1,
            prompt=prompt,
            output=output,
            scores={},
        ),
    ]
    store = _make_store(tmp_path, "rescore-cell", "abl-bind-also-plausible", records)

    cases_by_id = {
        "g-hit": _candidates_case("g-hit", stat_vars=[dcid_hit, dcid_uncached]),
        "g-all-unscorable": _candidates_case("g-all-unscorable", stat_vars=[dcid_only_uncached]),
    }

    ro_graph = report.open_read_only_graph_cache(graph_path)
    try:
        enriched = report.rescore_cell(store, graph=ro_graph, cases_by_id=cases_by_id)
    finally:
        ro_graph.close()

    by_case = {r["case_id"]: r for r in enriched}
    # The uncached DCID is excluded from the recall denominator.
    assert by_case["g-hit"]["scores"]["expected_alternate_recall"] == pytest.approx(1.0)
    # Every alternate missed the cache, so the record is unscorable.
    assert by_case["g-all-unscorable"]["scores"]["expected_alternate_recall"] == "unscorable"

    scores = store.read_scores()
    assert scores["expected_alternate_recall_scored"] == 1
    assert scores["expected_alternate_recall_mean"] == pytest.approx(1.0)
    assert scores["expected_alternate_recall_unscorable"] == 1

    table = report.build_metric_table_from_records(
        cell_id=store.manifest.cell_id, rung_name="abl-bind-also-plausible", records=enriched
    )
    assert table.aggregate["expected_alternate_recall_mean"] == pytest.approx(1.0)
    assert table.aggregate["expected_alternate_recall_scored"] == 1
    assert table.aggregate["expected_alternate_recall_unscorable"] == 1


def test_rescore_cell_persists_scores_via_finish(tmp_path: Path):
    records = [
        _bind_record(
            case_id="g1",
            leg=0,
            seed=1,
            prompt="p",
            output={"also_plausible": []},
            scores={},
        ),
    ]
    store = _make_store(tmp_path, "rescore-noop", "abl-bind-x", records)
    graph_path = tmp_path / "unused-graph.sqlite"
    _seed_graph_cache(graph_path, cached_dcid="unused", arcs=_HIT_ARCS)
    ro_graph = report.open_read_only_graph_cache(graph_path)
    try:
        report.rescore_cell(
            store,
            graph=ro_graph,
            cases_by_id={"g1": _candidates_case("g1", stat_vars=[])},
        )
    finally:
        ro_graph.close()

    # Rescoring finishes the cell even when no expected StatVars exist.
    assert store.is_complete()
    assert "n_calls" in store.read_scores()
