"""Tests recalled labels, representative scores, and the confirm-slot reserve."""

from __future__ import annotations

import pytest

from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import STANDARD_SHAPE_LABEL, ShapeDraft
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.fixtures import FakeGraph


class TestDeriveShapesLabelScoreSplit:
    """A recalled member supplies the label.

    Expansion members can share its group or form their own."""

    _TOPIC = "dc/topic/GDP"

    def _gdp_graph(self) -> FakeGraph:
        nodes: dict = {
            self._TOPIC: {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": "Expand_B1", "types": ["StatisticalVariable"]},
                            {"dcid": "Expand_B2", "types": ["StatisticalVariable"]},
                            {"dcid": "Expand_C1", "types": ["StatisticalVariable"]},
                            {"dcid": "Expand_C2", "types": ["StatisticalVariable"]},
                        ]
                    }
                }
            },
            "GrowthRate_SV": {
                "label": "growth rate",
                "arcs": _std_sv_arcs("Country", "growthRate", "GDP growth rate"),
            },
            "Aggregate_SV": {
                "label": "aggregate",
                "arcs": _std_sv_arcs(
                    "Country", "grossDomesticProduction", "Nominal gross domestic product"
                ),
            },
            "Expand_B1": {
                "label": "expand b1",
                "arcs": _std_sv_arcs(
                    "Country",
                    "grossDomesticProduction",
                    "GDP of the Administrative and Support and Waste Management Services Industry",
                ),
            },
            "Expand_B2": {
                "label": "expand b2",
                "arcs": _std_sv_arcs(
                    "Country", "grossDomesticProduction", "GDP of the Manufacturing Industry"
                ),
            },
            "Expand_C1": {
                "label": "expand c1",
                "arcs": _std_sv_arcs(
                    "Country",
                    "grossValueAdded",
                    "Gross value added in agriculture, forestry and fishing sector "
                    "at current prices",
                ),
            },
            "Expand_C2": {
                "label": "expand c2",
                "arcs": _std_sv_arcs(
                    "Country", "grossValueAdded", "Gross value added in manufacturing sector"
                ),
            },
        }
        return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

    def _derive(self) -> list[ShapeDraft]:
        graph = self._gdp_graph()
        return derive_shapes(
            confirmed_svs=[self._TOPIC, "GrowthRate_SV", "Aggregate_SV"],
            graph=graph,
            sv_scores={self._TOPIC: 1.0, "GrowthRate_SV": 0.7898, "Aggregate_SV": 0.7883},
            sv_types={self._TOPIC: "Topic", "GrowthRate_SV": "SV", "Aggregate_SV": "SV"},
        )

    def _shape_with(self, shapes: list[ShapeDraft], sv_dcid: str) -> ShapeDraft:
        return next(s for s in shapes if sv_dcid in (s.sv_arc_facts or {}))

    def test_mixed_group_takes_real_members_label(self):
        """The recalled aggregate supplies the mixed group's label."""
        shapes = self._derive()
        aggregate_shape = self._shape_with(shapes, "Aggregate_SV")
        assert aggregate_shape.label == "Nominal gross domestic product"

    def test_mixed_group_representative_score_unchanged(self):
        """The group's score is the recalled aggregate's 0.7883."""
        shapes = self._derive()
        aggregate_shape = self._shape_with(shapes, "Aggregate_SV")
        assert aggregate_shape.representative_score == pytest.approx(0.7883)

    def test_pure_expansion_group_keeps_expanded_label_and_score(self):
        """A pure-expansion group keeps its label and has no score."""
        shapes = self._derive()
        c_shape = self._shape_with(shapes, "Expand_C1")
        assert "Expand_C2" in (c_shape.sv_arc_facts or {})
        assert c_shape.label == (
            "Gross value added in agriculture, forestry and fishing sector at current prices"
        )
        assert c_shape.representative_score is None

    def test_name_arc_fallback_never_reaches_generic_placeholder(self):
        """A recalled member without a name falls back to an expanded sibling's name."""
        nodes: dict = {
            self._TOPIC: {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [{"dcid": "Expand_D1", "types": ["StatisticalVariable"]}]
                    }
                }
            },
            "OtherReal_SV": {
                "label": "other real",
                "arcs": _std_sv_arcs("Country", "otherMetric", "Other Metric"),
            },
            "RealNoName_SV": {
                "label": "no name",
                "arcs": _std_sv_arcs("Country", "unnamedMetric"),
            },
            "Expand_D1": {
                "label": "expand d1",
                "arcs": _std_sv_arcs("Country", "unnamedMetric", "Expanded Metric Name"),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(
            confirmed_svs=[self._TOPIC, "OtherReal_SV", "RealNoName_SV"],
            graph=graph,
            sv_scores={self._TOPIC: 1.0, "OtherReal_SV": 0.90, "RealNoName_SV": 0.40},
            sv_types={self._TOPIC: "Topic", "OtherReal_SV": "SV", "RealNoName_SV": "SV"},
        )
        shape = self._shape_with(shapes, "RealNoName_SV")
        assert "Expand_D1" in (shape.sv_arc_facts or {})
        assert shape.label == "Expanded Metric Name"


class TestDeriveShapesReserveSlots:
    """The reserve protects high-scoring recalled candidates from wide topic expansions."""

    _TOPIC = "dc/topic/Wide"

    def test_reserve_rescues_boundary_candidate(self):
        """The reserve keeps its boundary candidate ahead of a wide expansion."""
        from search.engine.config import settings

        reserve = settings().min_recall_confirm_slots
        assert reserve >= 2, "fixture needs room for the boundary candidate plus one higher real"
        cap = settings().max_confirm_candidates

        nodes: dict = {
            self._TOPIC: {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": f"Expand_{i}", "types": ["StatisticalVariable"]}
                            for i in range(cap + 5)
                        ]
                    }
                }
            }
        }
        for i in range(cap + 5):
            nodes[f"Expand_{i}"] = {"label": "e", "arcs": _std_sv_arcs("Country", "expanded")}

        confirmed_svs = [self._TOPIC]
        sv_scores: dict[str, float] = {self._TOPIC: 1.0}
        sv_types: dict[str, str] = {self._TOPIC: "Topic"}

        # reserve - 1 real candidates score above the expansion stamp; Target_SV
        # is the reserve-th highest real score, just below them.
        for i in range(reserve - 1):
            dcid = f"HighReal_{i}"
            nodes[dcid] = {"label": "h", "arcs": _std_sv_arcs("Country", f"high_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.95 - i * 0.01
        nodes["Target_SV"] = {"label": "t", "arcs": _std_sv_arcs("Country", "target")}
        confirmed_svs.append("Target_SV")
        sv_scores["Target_SV"] = 0.95 - reserve * 0.01

        # More real candidates than the cap in total, scoring well below the
        # stamp -- proves "more than CAP real candidates" without competing
        # for the reserved slots.
        for i in range(cap + 1):
            dcid = f"LowReal_{i}"
            nodes[dcid] = {"label": "l", "arcs": _std_sv_arcs("Household", f"low_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.10 - i * 0.0001

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(
            confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores, sv_types=sv_types
        )
        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        assert "Target_SV" in all_members

    def test_reserve_disabled_a_recalled_candidate_still_survives(self, search_settings):
        """Target_SV ranks above unscored expansions when the reserve is zero."""
        from search.engine.config import settings

        search_settings(min_recall_confirm_slots=0)
        cap = settings().max_confirm_candidates

        nodes: dict = {
            self._TOPIC: {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": f"Expand_{i}", "types": ["StatisticalVariable"]}
                            for i in range(cap + 5)
                        ]
                    }
                }
            }
        }
        for i in range(cap + 5):
            nodes[f"Expand_{i}"] = {"label": "e", "arcs": _std_sv_arcs("Country", "expanded")}

        confirmed_svs = [self._TOPIC]
        sv_scores: dict[str, float] = {self._TOPIC: 1.0}
        sv_types: dict[str, str] = {self._TOPIC: "Topic"}
        nodes["Target_SV"] = {"label": "t", "arcs": _std_sv_arcs("Country", "target")}
        confirmed_svs.append("Target_SV")
        sv_scores["Target_SV"] = 0.50
        for i in range(cap + 1):
            dcid = f"LowReal_{i}"
            nodes[dcid] = {"label": "l", "arcs": _std_sv_arcs("Household", f"low_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.10 - i * 0.0001

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(
            confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores, sv_types=sv_types
        )
        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        assert "Target_SV" in all_members

    def test_reserve_changes_nothing_once_expansion_members_are_unscored(self, search_settings):
        """Unscored expansions yield the same confirmed set with reserve enabled or disabled."""
        from search.engine.config import settings

        cap = settings().max_confirm_candidates

        nodes: dict = {
            self._TOPIC: {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": f"Expand_{i}", "types": ["StatisticalVariable"]}
                            for i in range(cap + 5)
                        ]
                    }
                }
            }
        }
        for i in range(cap + 5):
            nodes[f"Expand_{i}"] = {"label": "e", "arcs": _std_sv_arcs("Country", "expanded")}

        confirmed_svs = [self._TOPIC]
        sv_scores: dict[str, float] = {self._TOPIC: 1.0}
        sv_types: dict[str, str] = {self._TOPIC: "Topic"}
        for i in range(cap + 1):
            dcid = f"Real_{i}"
            nodes[dcid] = {"label": "r", "arcs": _std_sv_arcs("Household", f"real_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.95 - i * 0.0001

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        def _members() -> set[str]:
            return {
                sv
                for s in derive_shapes(
                    confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores, sv_types=sv_types
                )
                for sv in (s.sv_arc_facts or {})
            }

        with_default_reserve = _members()
        search_settings(min_recall_confirm_slots=0)
        assert _members() == with_default_reserve

    @pytest.mark.parametrize(
        "rivals",
        [
            pytest.param(0, id="no-rival"),
            # Rivals recall added past its cut are confirmed past the cap, so the top `cap`
            # of the other candidates stand beside them.
            pytest.param(2, id="rivals-past-the-cap"),
        ],
    )
    def test_reserve_is_noop_without_expansion(self, search_settings, rivals: int):
        """Without expansion, the cap keeps the top-scoring candidates, and every rival."""
        from search.engine.config import settings

        cap = settings().max_confirm_candidates
        n = cap + 5
        nodes: dict = {}
        confirmed_svs = []
        sv_scores: dict[str, float] = {}
        for i in range(n):
            dcid = f"Real_{i}"
            nodes[dcid] = {"label": "r", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 1.0 - i * 0.001
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        rival_svs = frozenset(f"Real_{i}" for i in range(rivals))

        def _members() -> set[str]:
            shapes = derive_shapes(
                confirmed_svs=confirmed_svs,
                graph=graph,
                sv_scores=sv_scores,
                rival_svs=rival_svs,
            )
            assert {sv for s in shapes if s.rival for sv in (s.sv_arc_facts or {})} == rival_svs
            return {sv for s in shapes for sv in (s.sv_arc_facts or {})}

        with_default_reserve = _members()
        search_settings(min_recall_confirm_slots=0)
        without_reserve = _members()

        assert with_default_reserve == without_reserve
        assert with_default_reserve == {f"Real_{i}" for i in range(cap + rivals)}

    def test_reserve_does_not_starve_expansion(self):
        """Expansion fills slots left after real candidates use the reserve."""
        from search.engine.config import settings

        reserve = settings().min_recall_confirm_slots
        cap = settings().max_confirm_candidates
        assert reserve > 0
        n_expanded = cap - reserve + 8  # comfortably more than cap - reserve

        nodes: dict = {
            self._TOPIC: {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": f"Expand_{i}", "types": ["StatisticalVariable"]}
                            for i in range(n_expanded)
                        ]
                    }
                }
            }
        }
        for i in range(n_expanded):
            nodes[f"Expand_{i}"] = {"label": "e", "arcs": _std_sv_arcs("Country", "expanded")}

        confirmed_svs = [self._TOPIC]
        sv_scores: dict[str, float] = {self._TOPIC: 0.99}
        sv_types: dict[str, str] = {self._TOPIC: "Topic"}
        # Exactly `reserve` real candidates -- all get reserved, leaving none
        # left over to compete for the remaining cap - reserve slots.
        for i in range(reserve):
            dcid = f"Real_{i}"
            nodes[dcid] = {"label": "r", "arcs": _std_sv_arcs("Country", f"real_{i}")}
            confirmed_svs.append(dcid)
            sv_scores[dcid] = 0.70 - i * 0.01

        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        shapes = derive_shapes(
            confirmed_svs=confirmed_svs, graph=graph, sv_scores=sv_scores, sv_types=sv_types
        )
        expanded_shape = next(s for s in shapes if "Expand_0" in (s.sv_arc_facts or {}))
        assert len(expanded_shape.sv_arc_facts or {}) == cap - reserve
        real_svs = {f"Real_{i}" for i in range(reserve)}
        all_members = {sv for s in shapes for sv in (s.sv_arc_facts or {})}
        assert real_svs <= all_members


class TestDeriveShapesSdgLabelUnchanged:
    """The topic-expanded SDG group has no recall score or recalled label."""

    def test_sdg_group_label_and_score_match_representative(self):
        from tests.engine._discover_fixtures import _sdg_taxonomy_graph

        graph = _sdg_taxonomy_graph()
        shapes = derive_shapes(
            confirmed_svs=["dc/topic/HIV"],
            graph=graph,
            sv_scores={"dc/topic/HIV": 0.90},
            sv_types={"dc/topic/HIV": "Topic"},
        )
        assert len(shapes) == 1
        shape = shapes[0]
        assert set(shape.sv_arc_facts or {}) == {
            "sdg/SH_HIV_INCD.SEX--F",
            "sdg/SH_HIV_INCD.SEX--M",
        }
        assert shape.representative_score is None
        assert shape.label == STANDARD_SHAPE_LABEL
