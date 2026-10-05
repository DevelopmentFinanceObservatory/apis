"""Pins multi-root Topic expansion across the four-hop SDG taxonomy and its depth and size limits."""

from __future__ import annotations

from search.engine.config import settings
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.topics import expand_topics
from tests.engine._discover_fixtures import _sdg_taxonomy_graph
from tests.fixtures import FakeGraph


def _std_sv_arcs(pop_type: str = "Person", meas_prop: str = "count") -> dict:
    """Minimal standard (non-dev-finance) SV arcs: no constraint properties."""
    return {
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


class TestExpandTopicsMultiHop:
    def test_three_hop_chain_reaches_leaves(self):
        """A root three container hops away from its leaves is fully reached
        within the default max_depth (BFS keeps descending through nested
        Topic/StatVarPeerGroup containers, not just one relevantVariable hop)."""
        graph = _sdg_taxonomy_graph()
        leaf_to_root = expand_topics(["dc/topic/HIV"], graph=graph)
        assert set(leaf_to_root) == {"sdg/SH_HIV_INCD.SEX--F", "sdg/SH_HIV_INCD.SEX--M"}
        assert all(root == "dc/topic/HIV" for root in leaf_to_root.values())

    def test_max_depth_bound_stops_short_of_leaves(self):
        """A two-hop limit stops before the four-hop chain reaches a leaf."""
        graph = _sdg_taxonomy_graph()
        # The HIV chain has four edges to its leaves.
        leaf_to_root = expand_topics(["dc/topic/HIV"], graph=graph, max_depth=2)
        assert leaf_to_root == {}

    def test_each_level_is_one_batched_call(self):
        """BFS issues one batch call per level, regardless of frontier size."""
        calls: list[list[str]] = []
        graph = _sdg_taxonomy_graph()
        orig = graph.node_arcs_batch

        def _counting_batch(dcids):
            calls.append(list(dcids))
            return orig(dcids)

        graph.node_arcs_batch = _counting_batch  # type: ignore[method-assign]
        expand_topics(["dc/topic/HIV"], graph=graph)
        # Four graph reads reach the leaves.
        assert len(calls) == 4

    def test_two_independent_roots_merge_batches_per_level(self):
        """Two roots share one batch per level, using 3 calls instead of 6."""
        nodes = {
            "dc/topic/Root_A": {
                "arcs": {
                    "relevantVariable": {"nodes": [{"dcid": "dc/topic/Mid_A", "types": ["Topic"]}]}
                }
            },
            "dc/topic/Mid_A": {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [{"dcid": "dc/svpg/Group_A", "types": ["StatVarPeerGroup"]}]
                    }
                }
            },
            "dc/svpg/Group_A": {
                "arcs": {
                    "member": {"nodes": [{"dcid": "Leaf_A", "types": ["StatisticalVariable"]}]}
                }
            },
            "dc/topic/Root_B": {
                "arcs": {
                    "relevantVariable": {"nodes": [{"dcid": "dc/topic/Mid_B", "types": ["Topic"]}]}
                }
            },
            "dc/topic/Mid_B": {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [{"dcid": "dc/svpg/Group_B", "types": ["StatVarPeerGroup"]}]
                    }
                }
            },
            "dc/svpg/Group_B": {
                "arcs": {
                    "member": {"nodes": [{"dcid": "Leaf_B", "types": ["StatisticalVariable"]}]}
                }
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        calls: list[list[str]] = []
        orig = graph.node_arcs_batch

        def _counting_batch(dcids):
            calls.append(list(dcids))
            return orig(dcids)

        graph.node_arcs_batch = _counting_batch  # type: ignore[method-assign]
        leaf_to_root = expand_topics(["dc/topic/Root_A", "dc/topic/Root_B"], graph=graph)
        assert len(calls) == 3
        assert leaf_to_root == {"Leaf_A": "dc/topic/Root_A", "Leaf_B": "dc/topic/Root_B"}

    def test_multiple_roots_are_independent_and_deduped(self):
        """Shared leaves appear once and keep the first root's attribution."""
        graph = _sdg_taxonomy_graph()
        # Both roots lead to the same HIV leaves.
        leaf_to_root = expand_topics(["dc/topic/sdg_3.3.1", "dc/topic/HIV"], graph=graph)
        assert set(leaf_to_root) == {"sdg/SH_HIV_INCD.SEX--F", "sdg/SH_HIV_INCD.SEX--M"}
        # First root in the argument list wins the attribution for the shared leaves.
        assert all(root == "dc/topic/sdg_3.3.1" for root in leaf_to_root.values())

    def test_oversized_root_declines_without_affecting_other_roots(self):
        """An oversized root is dropped while a smaller root still contributes leaves."""
        huge_nodes = {
            "dc/topic/Huge": {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": f"Huge_Member_{i}", "types": ["StatisticalVariable"]}
                            for i in range(5)
                        ]
                    }
                }
            },
        }
        graph = _sdg_taxonomy_graph()
        graph._nodes.update(huge_nodes)
        leaf_to_root = expand_topics(["dc/topic/Huge", "dc/topic/HIV"], graph=graph, max_members=2)
        assert "Huge_Member_0" not in leaf_to_root
        assert set(leaf_to_root) == {"sdg/SH_HIV_INCD.SEX--F", "sdg/SH_HIV_INCD.SEX--M"}

    def test_falls_back_to_dcid_substring_when_types_absent(self):
        """DCID prefixes classify containers when fixtures omit inline types."""
        nodes = {
            "dc/topic/Root": {"arcs": {"relevantVariable": {"nodes": [{"dcid": "dc/svpg/Group"}]}}},
            "dc/svpg/Group": {"arcs": {"member": {"nodes": [{"dcid": "Plain_Leaf_SV"}]}}},
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        leaf_to_root = expand_topics(["dc/topic/Root"], graph=graph)
        assert leaf_to_root == {"Plain_Leaf_SV": "dc/topic/Root"}

    def test_leaves_sorted_least_constrained_first(self):
        """Leaves sort by constraint count so less-constrained members come first."""
        nodes = {
            "dc/topic/Root": {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": "Base.AGE--X__SEX--F", "types": ["StatisticalVariable"]},
                            {"dcid": "Base", "types": ["StatisticalVariable"]},
                            {"dcid": "Base.SEX--F", "types": ["StatisticalVariable"]},
                        ]
                    }
                }
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        leaf_to_root = expand_topics(["dc/topic/Root"], graph=graph)
        assert list(leaf_to_root) == ["Base", "Base.SEX--F", "Base.AGE--X__SEX--F"]


class TestDeriveShapesMultiRootGate:
    """derive_shapes' near-top-score root selection, exercised end-to-end."""

    def test_bounded_by_max_roots(self):
        """Only the top `topic_expand_max_roots` candidates are expanded."""
        n_topics = settings().topic_expand_max_roots + 2
        nodes = {}
        confirmed_svs = []
        sv_scores = {}
        sv_types = {}
        for i in range(n_topics):
            topic = f"dc/topic/Topic_{i}"
            member = f"Member_{i}"
            nodes[topic] = {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [{"dcid": member, "types": ["StatisticalVariable"]}]
                    }
                }
            }
            nodes[member] = {"label": "m", "arcs": _std_sv_arcs("Person", f"count_{i}")}
            confirmed_svs.append(topic)
            # Every topic qualifies by score; only the root cap excludes candidates.
            sv_scores[topic] = 0.90 - (i * 0.001)
            sv_types[topic] = "Topic"
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(
            confirmed_svs=confirmed_svs,
            graph=graph,
            sv_scores=sv_scores,
            sv_types=sv_types,
        )
        expanded_members = {
            next(iter(s.sv_arc_facts or {}))
            for s in shapes
            if next(iter(s.sv_arc_facts or {}), "").startswith("Member_")
        }
        assert len(expanded_members) == settings().topic_expand_max_roots
        # The cap keeps the highest-scoring roots.
        assert expanded_members == {f"Member_{i}" for i in range(settings().topic_expand_max_roots)}

    def test_wide_non_topic_pool_ranks_expansion_members_behind_every_measured_candidate(self):
        """Expanded members carry no score of their own, so the pool order
        places them behind every measured candidate: a non-topic pool that
        alone fills settings().max_confirm_candidates leaves no slot for expansion,
        and the expansion's own members never take one from a measured
        candidate. With room to spare, the expansion's member confirms."""

        def members(n_measured: int) -> set[str]:
            nodes = {
                "dc/topic/Root": {
                    "arcs": {
                        "relevantVariable": {
                            "nodes": [{"dcid": "Target_Member", "types": ["StatisticalVariable"]}]
                        }
                    }
                },
                "Target_Member": {"label": "t", "arcs": _std_sv_arcs("Person", "target")},
            }
            confirmed_svs = ["dc/topic/Root"]
            sv_scores = {"dc/topic/Root": 0.80}
            for i in range(n_measured):
                dcid = f"Measured_{i}"
                nodes[dcid] = {"label": "m", "arcs": _std_sv_arcs("Household", f"measured_{i}")}
                confirmed_svs.append(dcid)
                sv_scores[dcid] = 0.79
            graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
            return {
                sv
                for s in derive_shapes(
                    confirmed_svs=confirmed_svs,
                    graph=graph,
                    sv_scores=sv_scores,
                    sv_types={"dc/topic/Root": "Topic"},
                )
                for sv in (s.sv_arc_facts or {})
            }

        with_room = members(settings().max_confirm_candidates - 2)
        assert "Target_Member" in with_room
        assert len(with_room) == settings().max_confirm_candidates - 1

        filled = members(settings().max_confirm_candidates)
        assert "Target_Member" not in filled
        assert len(filled) == settings().max_confirm_candidates
        assert not any(sv == "Target_Member" for sv in filled)
