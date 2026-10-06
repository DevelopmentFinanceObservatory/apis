"""Tests candidate reserves, score thresholds, backfill, and batched graph reads."""

from __future__ import annotations

from collections import defaultdict

import pytest

from code_index.shared import get_index
from code_index.types import StatVarRow
from search.engine.config import settings
from search.engine.retrieval.derive import _index_group_keys, derive_shapes
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.fixtures import FakeGraph


def _financing_arcs() -> dict:
    """Builds a flow StatVar with financing-provider and recipient roles."""
    return {
        "populationType": {"nodes": [{"dcid": "CreditorDebtorFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "flowAmount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {
            "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
        },
    }


class _ArcsBatchCountingGraph:
    """Records arc and role batch arguments while delegating graph calls."""

    def __init__(self, inner: FakeGraph):
        self._inner = inner
        self.node_arcs_batch_calls: list[list[str]] = []
        self.node_observation_properties_batch_calls: list[list[str]] = []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        self.node_arcs_batch_calls.append(list(dcids))
        return self._inner.node_arcs_batch(dcids, deadline=deadline)

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        self.node_observation_properties_batch_calls.append(list(dcids))
        return self._inner.node_observation_properties_batch(dcids)


class _MisreadRolesGraph:
    """Returns empty role data for selected StatVars."""

    def __init__(self, inner: FakeGraph, misread_dcids: set[str]):
        self._inner = inner
        self._misread_dcids = misread_dcids

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        result = self._inner.node_observation_properties_batch(dcids)
        return {
            dcid: (() if dcid in self._misread_dcids else roles) for dcid, roles in result.items()
        }


class TestShapeClassFloorRescuesStandardCandidate:
    """The class floor protects viable standard candidates from a role-bearing flood."""

    def test_lower_scored_standard_candidate_still_confirms(self):
        cap = settings().max_confirm_candidates
        n_role_bearing = cap + 10

        nodes: dict = {}
        confirmed_svs: list[str] = []
        sv_scores: dict[str, float] = {}
        for i in range(n_role_bearing):
            dcid = f"Crs_{i}"
            nodes[dcid] = {"label": f"crs {i}", "arcs": _financing_arcs()}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.90 - i * 0.001

        standard_sv = "Count_Person_LowScore"
        nodes[standard_sv] = {"label": "low", "arcs": _std_sv_arcs("Person", "count")}
        confirmed_svs.append(standard_sv)
        sv_scores[standard_sv] = settings().unknown_score_threshold + 0.05

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores)

        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        assert standard_sv in all_members


class TestShapeClassFloorNoOpWhenPoolFitsCap:
    """A pool within the cap needs one arc read and no role read."""

    def test_pool_within_cap_makes_one_batch_call_over_the_whole_pool(self):
        n = settings().max_confirm_candidates - 3
        nodes: dict = {}
        confirmed_svs: list[str] = []
        for i in range(n):
            dcid = f"Real_{i}"
            nodes[dcid] = {"label": "r", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
            confirmed_svs.append(dcid)

        inner = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        graph = _ArcsBatchCountingGraph(inner)

        shapes = derive_shapes(confirmed_svs=confirmed_svs, graph=graph)

        assert len(graph.node_arcs_batch_calls) == 1
        assert sorted(graph.node_arcs_batch_calls[0]) == sorted(confirmed_svs)
        assert graph.node_observation_properties_batch_calls == []
        assert {sv for s in shapes for sv in (s.sv_arc_facts or {})} == set(confirmed_svs)


class TestShapeClassFloorUnconfirmableDoesNotConsumeSlot:
    """An unconfirmable top candidate is dropped and backfilled with one graph read."""

    def test_unconfirmable_high_scoring_dcid_is_backfilled_from_the_next_ranked_member(self):
        cap = settings().max_confirm_candidates

        nodes: dict = {}
        confirmed_svs: list[str] = []
        sv_scores: dict[str, float] = {}
        for i in range(cap):
            dcid = f"Real_{i}"
            nodes[dcid] = {"label": "r", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.90 - i * 0.001

        phantom = "Phantom_NotInGraph"
        confirmed_svs.append(phantom)
        sv_scores[phantom] = 0.999  # highest score in the pool, but unconfirmable

        inner = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        graph = _ArcsBatchCountingGraph(inner)
        shapes = derive_shapes(confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores)

        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        assert phantom not in all_members
        assert all_members == set(confirmed_svs) - {phantom}
        assert len(all_members) == cap

        # The confirm read is cap-sized, then a one-member backfill read runs.
        assert [len(c) for c in graph.node_arcs_batch_calls] == [cap, 1]


class TestRolesReadCoversOnlyViableMembers:
    """Role classification reads only candidates above the score threshold."""

    def test_node_observation_properties_batch_covers_only_at_or_above_threshold_scores(self):
        cap = settings().max_confirm_candidates
        n_above = cap + 5
        n_below = 10

        nodes: dict = {}
        confirmed_svs: list[str] = []
        sv_scores: dict[str, float] = {}
        above_svs: set[str] = set()
        for i in range(n_above):
            dcid = f"Above_{i}"
            nodes[dcid] = {"label": "std", "arcs": _std_sv_arcs("Country", f"above_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = settings().unknown_score_threshold + 0.05 - i * 0.0001
            above_svs.add(dcid)
        for i in range(n_below):
            dcid = f"Below_{i}"
            nodes[dcid] = {"label": "std", "arcs": _std_sv_arcs("Country", f"below_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = settings().unknown_score_threshold - 0.01 - i * 0.0001

        inner = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        graph = _ArcsBatchCountingGraph(inner)

        derive_shapes(confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores)

        assert len(graph.node_observation_properties_batch_calls) == 1
        assert set(graph.node_observation_properties_batch_calls[0]) == above_svs


class TestRoleBearingMisreadAsStandardIsNotPromoted:
    """A failed role read classifies a role-bearing candidate as standard."""

    def test_misread_role_bearing_member_not_promoted_by_the_role_bearing_floor(self):
        cap = settings().max_confirm_candidates
        n_standard = cap + 10

        nodes: dict = {}
        confirmed_svs: list[str] = []
        sv_scores: dict[str, float] = {}
        for i in range(n_standard):
            dcid = f"Std_{i}"
            nodes[dcid] = {"label": "std", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.90 - i * 0.0001

        role_bearing_svs = [f"Crs_{i}" for i in range(5)]
        for i, dcid in enumerate(role_bearing_svs):
            nodes[dcid] = {"label": "crs", "arcs": _financing_arcs()}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = settings().unknown_score_threshold + 0.05 - i * 0.0001

        inner = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        graph = _MisreadRolesGraph(inner, misread_dcids=set(role_bearing_svs))

        shapes = derive_shapes(confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores)

        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        naive_top = set(sorted(confirmed_svs, key=lambda sv: sv_scores[sv], reverse=True)[:cap])
        assert all_members == naive_top
        assert not any(sv.startswith("Crs_") for sv in all_members)


class TestShapeClassFloorCapInvariant:
    """The recall reserve and class floors together stay within the configured cap.

    A small cap makes the two floors sum past it, a config the shipped defaults never reach.
    """

    def test_confirmed_count_never_exceeds_cap_under_crowded_config(self, search_settings):
        nodes: dict = {}
        confirmed_svs: list[str] = []
        sv_scores: dict[str, float] = {}
        for i in range(8):
            dcid = f"Crs_{i}"
            nodes[dcid] = {"label": "crs", "arcs": _financing_arcs()}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.90 - i * 0.01
        for i in range(8):
            dcid = f"Std_{i}"
            nodes[dcid] = {"label": "std", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = settings().unknown_score_threshold + 0.05 - i * 0.0001

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        search_settings(
            max_confirm_candidates=10,
            min_recall_confirm_slots=8,
            min_shape_class_confirm_slots=8,
        )
        shapes = derive_shapes(confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores)

        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        assert len(all_members) == 10

        recall_reserve_top = {f"Crs_{i}" for i in range(8)}
        assert recall_reserve_top <= all_members

        standard_members = {sv for sv in all_members if sv.startswith("Std_")}
        assert standard_members == {"Std_0", "Std_1"}


class TestShapeClassFloorPinsExactTopN:
    """The floor preserves the top-N standard candidates.

    A floor collapsed to one would let the role-bearing flood evict all but the top candidate.
    """

    def test_top_scored_standard_candidates_survive_not_just_one(self):
        cap = settings().max_confirm_candidates
        floor = settings().min_shape_class_confirm_slots
        n_role_bearing = cap + 50
        n_standard = floor + 12

        nodes: dict = {}
        confirmed_svs: list[str] = []
        sv_scores: dict[str, float] = {}
        for i in range(n_role_bearing):
            dcid = f"Crs_{i}"
            nodes[dcid] = {"label": "crs", "arcs": _financing_arcs()}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.90 - i * 0.0001

        standard_svs = [f"Std_{i}" for i in range(n_standard)]
        for i, dcid in enumerate(standard_svs):
            nodes[dcid] = {"label": "std", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = settings().unknown_score_threshold + 0.05 - i * 0.0001

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores)

        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        top_standard = set(sorted(standard_svs, key=lambda sv: sv_scores[sv], reverse=True)[:floor])
        assert top_standard <= all_members
        assert not (set(standard_svs) - top_standard) & all_members


class TestShapeClassFloorSkipsBelowThresholdClass:
    """Below-threshold role-bearing members receive no reserved slots."""

    def test_below_threshold_role_bearing_family_not_promoted_past_cap_line(self):
        cap = settings().max_confirm_candidates
        n_standard = cap + 10

        nodes: dict = {}
        confirmed_svs: list[str] = []
        sv_scores: dict[str, float] = {}
        for i in range(n_standard):
            dcid = f"Std_{i}"
            nodes[dcid] = {"label": "std", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.90 - i * 0.0001

        n_role_bearing = 30
        for i in range(n_role_bearing):
            dcid = f"Crs_{i}"
            nodes[dcid] = {"label": "crs", "arcs": _financing_arcs()}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = settings().unknown_score_threshold - 0.09 - i * 0.001

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores)

        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        naive_top = set(sorted(confirmed_svs, key=lambda sv: sv_scores[sv], reverse=True)[:cap])
        assert all_members == naive_top
        assert not any(sv.startswith("Crs_") for sv in all_members)


class TestShapeClassFloorRescuesAboveThresholdRoleBearingFromWideStandardLattice:
    """Above-threshold role-bearing members survive a wide standard-class flood."""

    def test_above_threshold_role_bearing_family_still_rescued(self):
        cap = settings().max_confirm_candidates
        n_standard = cap + 10

        nodes: dict = {}
        confirmed_svs: list[str] = []
        sv_scores: dict[str, float] = {}
        for i in range(n_standard):
            dcid = f"Std_{i}"
            nodes[dcid] = {"label": "std", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.90 - i * 0.0001

        role_bearing_svs = [f"Crs_{i}" for i in range(5)]
        for i, dcid in enumerate(role_bearing_svs):
            nodes[dcid] = {"label": "crs", "arcs": _financing_arcs()}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = settings().unknown_score_threshold + 0.05 - i * 0.0001

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores)

        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        assert set(role_bearing_svs) <= all_members


def _index_group_rows(count_a: int, count_b: int) -> tuple[list[StatVarRow], list[StatVarRow]]:
    """Rows of the two largest packaged-index family groups whose five-tuples have no
    denominator, `count_a` from the larger and `count_b` from the other."""
    rows_by_dcid = {row.dcid: row for row in get_index().statvars}
    groups: dict[object, list[StatVarRow]] = defaultdict(list)
    for dcid, key in _index_group_keys(get_index()).items():
        if key[0].measurement_denominator is None:
            groups[key].append(rows_by_dcid[dcid])
    larger, other = sorted(groups.values(), key=len, reverse=True)[:2]
    return larger[:count_a], other[:count_b]


def _index_row_arcs(row: StatVarRow) -> dict:
    """Graph arcs of a packaged-index row: its five-tuple, roles, and constraints."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }
    if row.measurement_qualifier is not None:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": row.measurement_qualifier}]}
    return arcs


class TestFamilySpreadOverTheCap:
    """Over the cap, one index family group confirms its top member before its others.

    Pool entries are `a<i>` (rows of one family group), `b<i>` (rows of another), `n<i>`
    (StatVars the index does not hold) and `p<i>` (StatVars absent from the graph), listed
    by descending score. `reads` are the arc reads in order: the capped confirm, then the
    backfill when a member is absent."""

    @pytest.mark.parametrize(
        ("cap", "pool", "reads"),
        [
            pytest.param(
                4,
                ["a0", "a1", "a2", "a3", "a4", "a5", "b0"],
                [["a0", "b0", "a1", "a2"]],
                id="surplus-members-yield-to-another-family",
            ),
            pytest.param(
                3,
                ["p0", "p1", "a0", "a1", "a2", "a3", "a4", "a5", "b0"],
                [["p0", "p1", "a0"], ["b0", "a1"]],
                id="backfill-draws-in-spread-order",
            ),
            pytest.param(
                4,
                ["a0", "n0", "a1", "n1", "a2", "n2", "n3", "b0"],
                [["a0", "n0", "n1", "n2"]],
                id="unindexed-members-keep-their-rank",
            ),
            pytest.param(
                10,
                ["a0", "a1", "a2", "n0", "b0"],
                [["a0", "a1", "a2", "n0", "b0"]],
                id="pool-within-cap-keeps-rank-order",
            ),
        ],
    )
    def test_confirm_reads(self, search_settings, cap, pool, reads):
        search_settings(
            max_confirm_candidates=cap, min_recall_confirm_slots=0, min_shape_class_confirm_slots=0
        )
        rows_a, rows_b = _index_group_rows(
            sum(name[0] == "a" for name in pool), sum(name[0] == "b" for name in pool)
        )
        rows = {f"a{i}": row for i, row in enumerate(rows_a)} | {
            f"b{i}": row for i, row in enumerate(rows_b)
        }
        dcids = {name: rows[name].dcid if name in rows else f"Unindexed_{name}" for name in pool}
        nodes = {
            dcids[name]: {
                "label": name,
                "arcs": (
                    _index_row_arcs(rows[name])
                    if name in rows
                    else _std_sv_arcs("Country", f"metric_{name}")
                ),
            }
            for name in pool
            if name[0] != "p"
        }
        sv_scores = {dcids[name]: 0.9 - rank * 0.01 for rank, name in enumerate(pool)}

        graph = _ArcsBatchCountingGraph(FakeGraph(nodes=nodes, obs={}, detect={}, resolve={}))
        shapes = derive_shapes(
            confirmed_svs=[dcids[name] for name in pool], graph=graph, sv_scores=sv_scores
        )

        assert graph.node_arcs_batch_calls == [[dcids[name] for name in read] for read in reads]
        confirmed = {name for read in reads for name in read if name[0] != "p"}
        members = {sv for shape in shapes for sv in (shape.sv_arc_facts or {})}
        assert members == {dcids[name] for name in confirmed}
