"""Tests topic classification, bounded expansion, and shape selection."""

from __future__ import annotations

import pytest

from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.topics import is_topic_candidate
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# Topic keep/expand decision (is_topic_candidate, derive_shapes)
# ---------------------------------------------------------------------------


class TestIsTopicCandidate:
    def test_type_of_topic_is_topic(self):
        assert is_topic_candidate("some/dcid", "Topic") is True

    def test_type_of_statistical_variable_is_not_topic(self):
        assert is_topic_candidate("some/dcid", "StatisticalVariable") is False

    def test_falls_back_to_dcid_substring_when_type_unknown(self):
        assert is_topic_candidate("dc/topic/Forest", None) is True
        assert is_topic_candidate("Count_Person", None) is False

    def test_type_of_takes_precedence_over_dcid_substring(self):
        # Authoritative typeOf wins even if the dcid happens to contain "/topic/".
        assert is_topic_candidate("dc/topic/Forest", "StatisticalVariable") is False


class TestDeriveShapesTopicDecision:
    """Tests topic keep, expand, and skip decisions through shape derivation."""

    _TOPIC = "dc/topic/Whatever"
    _OTHER_SV = "Some_Other_SV"
    _MEMBER = "Topic_Member_SV"

    def _graph(self, *, member_count: int = 1) -> FakeGraph:
        nodes: dict = {
            self._TOPIC: {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [{"dcid": f"{self._MEMBER}_{i}"} for i in range(member_count)]
                    }
                }
            },
            self._OTHER_SV: {"label": "other", "arcs": _std_sv_arcs("Household", "income")},
        }
        for i in range(member_count):
            nodes[f"{self._MEMBER}_{i}"] = {
                "label": "member",
                "arcs": _std_sv_arcs("Person", "count"),
            }
        return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

    def test_single_dominant_topic_is_expanded(self):
        graph = self._graph(member_count=1)
        shapes = derive_shapes(
            confirmed_svs=[self._TOPIC, self._OTHER_SV],
            graph=graph,
            sv_scores={self._TOPIC: 0.90, self._OTHER_SV: 0.50},
            sv_types={self._TOPIC: "Topic", self._OTHER_SV: "StatisticalVariable"},
        )
        assert len(shapes) == 2
        member_shape = next(s for s in shapes if self._MEMBER in next(iter(s.sv_arc_facts or {})))
        assert member_shape.representative_score is None

    def test_topic_within_margin_of_top_score_is_expanded(self):
        """A Topic near the pool's top score expands despite other candidates."""
        graph = self._graph(member_count=1)
        shapes = derive_shapes(
            confirmed_svs=[self._TOPIC, self._OTHER_SV],
            graph=graph,
            sv_scores={self._TOPIC: 0.72, self._OTHER_SV: 0.70},  # topic leads by 0.02
            sv_types={self._TOPIC: "Topic", self._OTHER_SV: "StatisticalVariable"},
        )
        assert len(shapes) == 2
        member_shape = next(s for s in shapes if self._MEMBER in next(iter(s.sv_arc_facts or {})))
        assert member_shape.representative_score is None

    def test_topic_far_behind_top_score_is_not_expanded(self):
        """A Topic outside the pool-best margin stays unexpanded."""
        graph = self._graph(member_count=1)
        shapes = derive_shapes(
            confirmed_svs=[self._TOPIC, self._OTHER_SV],
            graph=graph,
            sv_scores={self._TOPIC: 0.60, self._OTHER_SV: 0.90},  # topic trails by 0.30
            sv_types={self._TOPIC: "Topic", self._OTHER_SV: "StatisticalVariable"},
        )
        assert len(shapes) == 1
        assert next(iter(shapes[0].sv_arc_facts or {})) == self._OTHER_SV

    def test_multiple_qualifying_topics_are_all_expanded(self):
        """Each qualifying Topic expands up to the root cap.

        An empty second root contributes zero shapes."""
        graph = self._graph(member_count=1)
        graph._nodes["dc/topic/AnotherOne"] = {"arcs": {"relevantVariable": {"nodes": []}}}
        shapes = derive_shapes(
            confirmed_svs=[self._TOPIC, "dc/topic/AnotherOne", self._OTHER_SV],
            graph=graph,
            sv_scores={self._TOPIC: 0.95, "dc/topic/AnotherOne": 0.93, self._OTHER_SV: 0.50},
            sv_types={
                self._TOPIC: "Topic",
                "dc/topic/AnotherOne": "Topic",
                self._OTHER_SV: "StatisticalVariable",
            },
        )
        assert len(shapes) == 2
        member_shape = next(s for s in shapes if self._MEMBER in next(iter(s.sv_arc_facts or {})))
        assert member_shape.representative_score is None

    def test_container_with_giant_fanout_is_not_expanded(self):
        """A topic above the member cap stays unexpanded through ``derive_shapes``."""
        from search.engine.config import settings

        graph = self._graph(member_count=settings().topic_expand_max_members + 1)
        shapes = derive_shapes(
            confirmed_svs=[self._TOPIC, self._OTHER_SV],
            graph=graph,
            sv_scores={self._TOPIC: 0.95, self._OTHER_SV: 0.50},
            sv_types={self._TOPIC: "Topic", self._OTHER_SV: "StatisticalVariable"},
        )
        assert len(shapes) == 1
        assert next(iter(shapes[0].sv_arc_facts or {})) == self._OTHER_SV

    def test_expansion_does_not_crowd_out_real_aggregate_candidate(self):
        """The 0.9084 aggregate keeps its score beside 45 unscored expanded members."""
        n_members = 45
        nodes: dict = {
            "dc/topic/GDP": {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": f"NAICS_Member_{i}", "types": ["StatisticalVariable"]}
                            for i in range(n_members)
                        ]
                    }
                }
            },
            "Aggregate_SV": {
                "label": "aggregate",
                "arcs": _std_sv_arcs("Country", "grossDomesticProduction"),
            },
        }
        for i in range(n_members):
            nodes[f"NAICS_Member_{i}"] = {
                "label": f"naics_{i}",
                "arcs": _std_sv_arcs("Country", f"naics_{i}"),
            }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(
            confirmed_svs=["dc/topic/GDP", "Aggregate_SV"],
            graph=graph,
            sv_scores={"dc/topic/GDP": 1.0, "Aggregate_SV": 0.9084},
            sv_types={"dc/topic/GDP": "Topic", "Aggregate_SV": "StatisticalVariable"},
        )
        aggregate_shape = next(
            (s for s in shapes if "Aggregate_SV" in (s.sv_arc_facts or {})), None
        )
        assert aggregate_shape is not None, "aggregate was crowded out of the confirm pool"
        assert aggregate_shape.representative_score == pytest.approx(0.9084)
        expanded_shapes = [s for s in shapes if s is not aggregate_shape]
        assert expanded_shapes
        assert all(s.representative_score is None for s in expanded_shapes)
