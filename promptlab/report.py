"""Build per-metric reports and rescore stored calls without LLM traffic.

Record shape identifies cell type. E2e records carry `checks` and `llm_calls`. Bind and
extract records carry `scores`. Aggregation reuses `promptlab.cells` and `promptlab.e2e_cell`.
Reports group metrics by seed for noise bands. Rescoring computes graph-dependent bind nomination
recall.

`--against` compares each metric with the ceiling's min/max across its seeds. Verdicts account
for metric direction. Informational counts retain values but omit verdicts.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dc_graph import ArcTarget
from graph_generation import UNKNOWN_STAMP, GenerationStamp
from promptlab.bindmenu import parse_menu
from promptlab.expectations import LegExpectation, leg_expectation
from promptlab.graphcache import CacheMiss, GraphCache
from promptlab.scoring import make_nomination_correctness
from promptlab.store import CellStore, NotComplete, StoreError
from search.engine.clients.graph import Detection, DetectLegs
from search.eval.cases import CHECK_NAMES, Case

_VERDICT_PASS = "PASS"
_VERDICT_INSIDE = "INSIDE-BAND"
_VERDICT_BELOW = "BELOW"

_HIGHER_IS_BETTER = "higher_is_better"
_LOWER_IS_BETTER = "lower_is_better"
_INFORMATIONAL = "informational"


class ReportError(Exception):
    """A cell cannot serve as a report input, or as a `--against` ceiling."""


# Classify every report metric before it can receive a `--against` verdict.
# Quality metrics are direction-aware. Informational counts omit verdicts.
# `_metric_direction` raises `ReportError` for unclassified metrics.
_METRIC_DIRECTION: dict[str, str] = {
    # Bind and extract diagnostics.
    "shape_election_match_rate": _HIGHER_IS_BETTER,
    "over_nomination_count_mean": _LOWER_IS_BETTER,  # more over-nomination is a regression
    "date_structure_match_rate": _HIGHER_IS_BETTER,
    "entity_dcid_match_rate": _HIGHER_IS_BETTER,
    "shape_recall_mean": _HIGHER_IS_BETTER,
    "variable_split_match_rate": _HIGHER_IS_BETTER,
    # Graph-dependent bind recall and its scored/unscorable counts.
    "expected_alternate_recall_mean": _HIGHER_IS_BETTER,
    "expected_alternate_recall_scored": _INFORMATIONAL,
    "expected_alternate_recall_unscorable": _INFORMATIONAL,
    # The pass rate of each eval check over the cases it scored.
    **{f"{name}_pass_rate": _HIGHER_IS_BETTER for name in CHECK_NAMES},
    # Fraction of same-case seed runs that disagree with the modal answer.
    "seed_flip_rate": _LOWER_IS_BETTER,
}


def _metric_direction(name: str) -> str:
    """Return `name`'s classification or raise `ReportError` if it is unlisted."""
    try:
        return _METRIC_DIRECTION[name]
    except KeyError:
        raise ReportError(
            f"metric {name!r} has no entry in promptlab.report._METRIC_DIRECTION -- "
            "classify it as higher-is-better, lower-is-better, or informational "
            "before it can enter a --against verdict table"
        ) from None


def open_cell(cells_root: Path, cell_id: str, *, allow_partial: bool) -> CellStore:
    """Open the cell, refusing partial data unless `allow_partial` is true."""
    try:
        store = CellStore.open(cells_root / cell_id)
    except StoreError as exc:
        raise ReportError(str(exc)) from exc
    if not allow_partial:
        try:
            store.require_complete()
        except NotComplete as exc:
            raise ReportError(f"{exc} -- pass --allow-partial to report on it anyway") from exc
    return store


def _is_e2e_record(record: Mapping[str, Any]) -> bool:
    return "checks" in record


def _e2e_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return the check pass rates and the seed-flip rate using `e2e_cell` helpers.

    Seed flips require multiple outputs per case, so per-seed subsets omit `seed_flip_rate`.
    """
    from promptlab.e2e_cell import _check_rates, seed_flip_rate_mean

    metrics = _check_rates(records)
    if (flip := seed_flip_rate_mean(records)) is not None:
        metrics["seed_flip_rate"] = flip
    return metrics


def _bind_or_extract_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate bind/extract diagnostics with `promptlab.cells`' helpers.

    Rescored bind records add alternate recall and scored/unscorable counts so its coverage is
    visible beside the mean.
    """
    from promptlab.cells import _aggregate_bind_scores, _aggregate_extract_scores

    bind_agg = _aggregate_bind_scores(records)
    extract_agg = _aggregate_extract_scores(records)
    metrics: dict[str, float] = {}
    if bind_agg["shape_election_scored"]:
        metrics["shape_election_match_rate"] = bind_agg["shape_election_match_rate"]
    if bind_agg["over_nomination_scored"]:
        metrics["over_nomination_count_mean"] = bind_agg["over_nomination_count_mean"]
    if extract_agg["date_structure_match_rate"] is not None:
        metrics["date_structure_match_rate"] = extract_agg["date_structure_match_rate"]
    if extract_agg["entity_dcid_match_rate"] is not None:
        metrics["entity_dcid_match_rate"] = extract_agg["entity_dcid_match_rate"]
    if extract_agg["shape_recall_scored"]:
        metrics["shape_recall_mean"] = extract_agg["shape_recall_mean"]
    if extract_agg["variable_split_match_rate"] is not None:
        metrics["variable_split_match_rate"] = extract_agg["variable_split_match_rate"]

    recall_records = [r for r in records if "expected_alternate_recall" in (r.get("scores") or {})]
    if recall_records:
        recalls = [
            r["scores"]["expected_alternate_recall"]
            for r in recall_records
            if isinstance(r["scores"]["expected_alternate_recall"], (int, float))
        ]
        unscorable = sum(
            1 for r in recall_records if r["scores"]["expected_alternate_recall"] == "unscorable"
        )
        if recalls:
            metrics["expected_alternate_recall_mean"] = sum(recalls) / len(recalls)
        # Keep counts beside the mean so its scored fraction stays visible.
        # Include counts even when the mean is absent.
        metrics["expected_alternate_recall_scored"] = len(recalls)
        metrics["expected_alternate_recall_unscorable"] = unscorable
    return metrics


def _metrics_for(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return (
        _e2e_metrics(records) if _is_e2e_record(records[0]) else _bind_or_extract_metrics(records)
    )


@dataclass(frozen=True)
class MetricTable:
    """Cell metrics aggregated over all records and per seed.

    `rung_name` lets JSON reports identify the unit an ablation cell removed.
    """

    cell_id: str
    rung_name: str
    n_seeds: int
    aggregate: dict[str, float]
    per_seed: dict[int, dict[str, float]]

    def band(self, metric: str) -> tuple[float, float] | None:
        """Return `metric`'s seed range, or `None` when every seed omits it."""
        values = [
            seed_metrics[metric]
            for seed_metrics in self.per_seed.values()
            if metric in seed_metrics
        ]
        return (min(values), max(values)) if values else None


def build_metric_table(store: CellStore) -> MetricTable:
    return build_metric_table_from_records(
        cell_id=store.manifest.cell_id,
        rung_name=store.manifest.rung.get("name", store.manifest.cell_id),
        records=store.call_records(),
    )


def build_metric_table_from_records(
    *, cell_id: str, rung_name: str, records: Sequence[Mapping[str, Any]]
) -> MetricTable:
    """Build a table from records, including `--rescore` enrichment.

    Rescored fields stay outside append-only `calls.jsonl`.
    """
    if not records:
        raise ReportError(f"cell {cell_id!r} has no call records to report")

    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for r in records:
        by_seed.setdefault(r["seed"], []).append(r)

    return MetricTable(
        cell_id=cell_id,
        rung_name=rung_name,
        n_seeds=len(by_seed),
        aggregate=_metrics_for(records),
        per_seed={seed: _metrics_for(recs) for seed, recs in sorted(by_seed.items())},
    )


def _bind_thinking_levels(records: Sequence[Mapping[str, Any]]) -> list[str | None]:
    """Return bind-call thinking levels from e2e and bind records.

    Extract records contribute none.
    """
    levels: list[str | None] = []
    for r in records:
        if "llm_calls" in r:
            levels.extend(c["thinking_level"] for c in r["llm_calls"] if c.get("stage") == "bind")
        elif r.get("stage") == "bind":
            levels.append(r.get("thinking_level"))
    return levels


def require_minimal_ceiling(store: CellStore) -> None:
    """Require every bind call to use `thinking_level == "minimal"`.

    A cell with no bind calls passes because it makes no claim about bind.
    """
    levels = _bind_thinking_levels(store.call_records())
    bad = sorted({str(level) for level in levels if level != "minimal"})
    if bad:
        raise ReportError(
            f"cell {store.manifest.cell_id!r} cannot serve as a --against ceiling: its "
            f"bind-stage calls are not all thinking_level='minimal' (found {bad!r})"
        )


def _verdict(value: float, band: tuple[float, float], *, direction: str) -> str:
    """Classify `value` against `band` in the metric's direction.

    Informational metrics are excluded by the renderers.
    """
    lo, hi = band
    if direction == _LOWER_IS_BETTER:
        if value > hi:
            return _VERDICT_BELOW
        if value < lo:
            return _VERDICT_PASS
        return _VERDICT_INSIDE
    if value < lo:
        return _VERDICT_BELOW
    if value > hi:
        return _VERDICT_PASS
    return _VERDICT_INSIDE


def render_markdown(tables: Sequence[MetricTable], *, against: MetricTable | None) -> str:
    lines: list[str] = []
    for table in tables:
        lines.append(f"## {table.cell_id}")
        lines.append("")
        lines.append(f"rung: `{table.rung_name}`, seeds: {table.n_seeds}")
        lines.append("")
        names = sorted(table.aggregate)
        if not names:
            lines.append("_no scored metrics_")
            lines.append("")
            continue
        if against is not None:
            lines.append("| metric | value | ceiling band | delta | verdict |")
            lines.append("| --- | --- | --- | --- | --- |")
            for name in names:
                value = table.aggregate[name]
                band = against.band(name)
                if band is None:
                    lines.append(f"| {name} | {value:.4f} | n/a | n/a | n/a |")
                    continue
                direction = _metric_direction(name)
                if direction == _INFORMATIONAL:
                    lines.append(f"| {name} | {value:.4f} | — | — | — |")
                    continue
                lo, hi = band
                delta = value - lo
                lines.append(
                    f"| {name} | {value:.4f} | [{lo:.4f}, {hi:.4f}] | {delta:+.4f} | "
                    f"{_verdict(value, band, direction=direction)} |"
                )
        else:
            lines.append("| metric | value |")
            lines.append("| --- | --- |")
            lines.extend(f"| {name} | {table.aggregate[name]:.4f} |" for name in names)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_json(tables: Sequence[MetricTable], *, against: MetricTable | None) -> dict[str, Any]:
    cells = []
    for table in tables:
        metrics: dict[str, Any] = {}
        for name, value in sorted(table.aggregate.items()):
            entry: dict[str, Any] = {"value": value}
            if against is not None:
                band = against.band(name)
                if band is not None:
                    direction = _metric_direction(name)
                    if direction != _INFORMATIONAL:
                        lo, hi = band
                        entry["ceiling_band"] = [lo, hi]
                        entry["delta"] = value - lo
                        entry["verdict"] = _verdict(value, band, direction=direction)
            metrics[name] = entry
        cells.append(
            {
                "cell_id": table.cell_id,
                "rung_name": table.rung_name,
                "n_seeds": table.n_seeds,
                "metrics": metrics,
            }
        )
    return {"cells": cells}


class _NoLiveGraph:
    """Sentinel client for a read-only cache.

    Any attempted live graph call raises `RuntimeError`.
    """

    def _unreachable(self, name: str) -> Any:
        raise RuntimeError(
            f"report --rescore's placeholder graph client was actually called "
            f"({name!r}) -- GraphCache(read_only=True) should never reach it"
        )

    def node_label(self, dcid: str) -> str | None:
        return self._unreachable("node_label")

    def un_availability(self) -> None:
        return self._unreachable("un_availability")

    def generation_stamp(self) -> GenerationStamp:
        """The unknown stamp: an offline graph has no generation."""
        return UNKNOWN_STAMP

    def source_key(self) -> str:
        """A fixed name: an offline graph reads no backend."""
        return "offline"

    def node_arcs(self, dcid: str) -> dict | None:
        return self._unreachable("node_arcs")

    def node_type(self, dcid: str) -> str | None:
        return self._unreachable("node_type")

    def default_facet_policy(self, *, stat_var: str, axis: str) -> None:
        """None, as the live client answers: the graph declares no default facet
        (`dc_graph.GraphClient.default_facet_policy`), and `GraphCache` passes this read
        through uncached."""
        return None

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return self._unreachable("node_arcs_many")

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        return self._unreachable("node_arc_targets_many")

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        return self._unreachable("node_names_many")

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        return self._unreachable("node_subjects_many")

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._unreachable("resolve_by_arc")

    def child_places_batch(
        self, parent_dcids: list[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._unreachable("child_places_batch")

    def detect_svs(self, query: str) -> Detection:
        return self._unreachable("detect_svs")

    def detect_legs(self, query: str) -> DetectLegs:
        return self._unreachable("detect_legs")

    def observation_facets(self, *, stat_var: str, entity: str, needs_dates: bool = False) -> list:
        return self._unreachable("observation_facets")

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        return self._unreachable("node_labels_batch")

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        return self._unreachable("node_types_batch")

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        return self._unreachable("node_arcs_batch")

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        return self._unreachable("node_observation_properties_batch")

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list]]:
        return self._unreachable("observation_facets_batch")

    def observation_read_groups(
        self, stat_vars: list[str], entities: list[str], needs_dates: bool = False
    ) -> list[list[str]]:
        return self._unreachable("observation_read_groups")

    def child_dcids(self, parent_dcid: str) -> list[str]:
        return self._unreachable("child_dcids")

    def enum_members(self, property_dcid: str) -> list[str]:
        return self._unreachable("enum_members")

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        return self._unreachable("currency_units")

    def entities_with_data(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        max_bytes: int | None = None,
    ) -> list[str]:
        return self._unreachable("entities_with_data")

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._unreachable("entities_with_data_by_variable")


def _resolve_one_alternate(
    dcid: str,
    *,
    evaluator: Any,
    output: Mapping[str, Any],
    expected: LegExpectation,
    menu: Sequence[Any],
) -> float | None:
    """Score one alternate in isolation so a `CacheMiss` leaves others unaffected.

    Return `None` for a miss or unscorable result. Return 1.0 or 0.0 for a score.
    """
    try:
        [evaluation] = evaluator(
            output=output, expected=dataclasses.replace(expected, stat_vars=(dcid,)), menu=menu
        )
    except CacheMiss:
        return None
    if isinstance(evaluation.value, str):  # the evaluator's own "unscorable"
        return None
    return float(evaluation.value)


def _rescore_bind_record(
    record: Mapping[str, Any], *, evaluator: Any, case: Case
) -> dict[str, Any]:
    expected = leg_expectation(case, record["leg"])
    scores = dict(record.get("scores") or {})
    if expected is None or expected.sole_status != "candidates" or not expected.stat_vars:
        return {**record, "scores": scores}

    _variable, menu = parse_menu(record["prompt"])
    hits = [
        hit
        for dcid in expected.stat_vars
        if (
            hit := _resolve_one_alternate(
                dcid,
                evaluator=evaluator,
                output=record["output"],
                expected=expected,
                menu=menu,
            )
        )
        is not None
    ]
    scores["expected_alternate_recall"] = (sum(hits) / len(hits)) if hits else "unscorable"
    return {**record, "scores": scores}


def open_read_only_graph_cache(path: Path) -> GraphCache:
    """Open the capture's `graph.sqlite` in read-only mode.

    Cache misses raise `CacheMiss` without live reads.
    """
    return GraphCache(_NoLiveGraph(), path=path, read_only=True)


def rescore_cell(
    store: CellStore, *, graph: GraphCache, cases_by_id: Mapping[str, Case]
) -> list[dict[str, Any]]:
    """Recompute `scores.json` from stored calls without LLM or live-graph traffic.

    `graph` must be the read-only cache over the cell's capture. `CellStore.finish` updates
    `scores.json` and `status.json`. The call log keeps its original records.

    Returns enriched records because bind-cell alternate recall is absent from the append-only
    log. Extract and e2e cells only recompute their stored scores.
    """
    records = list(store.call_records())
    if not records:
        raise ReportError(f"cell {store.manifest.cell_id!r} has no call records to rescore")

    if _is_e2e_record(records[0]) or records[0].get("stage") != "bind":
        store.finish(_metrics_for(records))
        return records

    from promptlab.cells import _aggregate_bind_scores

    evaluator = make_nomination_correctness(graph)
    enriched: list[dict[str, Any]] = []
    for record in records:
        case = cases_by_id.get(record["case_id"])
        if case is None:
            raise ReportError(f"no case supplied for captured case {record['case_id']!r}")
        enriched.append(_rescore_bind_record(record, evaluator=evaluator, case=case))

    scores = _aggregate_bind_scores(enriched)
    recalls = [
        r["scores"]["expected_alternate_recall"]
        for r in enriched
        if isinstance(r.get("scores", {}).get("expected_alternate_recall"), (int, float))
    ]
    unscorable = sum(
        1 for r in enriched if r.get("scores", {}).get("expected_alternate_recall") == "unscorable"
    )
    scores["expected_alternate_recall_mean"] = (sum(recalls) / len(recalls)) if recalls else None
    scores["expected_alternate_recall_scored"] = len(recalls)
    scores["expected_alternate_recall_unscorable"] = unscorable
    store.finish(scores)
    return enriched
