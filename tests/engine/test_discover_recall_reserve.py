"""Tests recalled labels, representative scores, and confirmation of every candidate."""

from __future__ import annotations

import time

import pytest

from dfo_platform.request_deadline import bounded_deadline
from search.engine.config import settings
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import STANDARD_SHAPE_LABEL, ShapeDraft
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.engine._fail_open_deadline import assert_fail_open_deadline
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


class TestDeriveShapesConfirmsEveryCandidate:
    """Every recalled candidate is confirmed, however many there are."""

    @pytest.mark.parametrize(
        "rivals",
        [
            pytest.param(0, id="no-rival"),
            pytest.param(2, id="rivals"),
        ],
    )
    def test_every_candidate_is_confirmed_and_probed_once_under_the_deadline(self, rivals: int):
        """Recall's `detect_k` candidates, four of them Topics whose leaves fill the member
        cap, and the rival rows past the cut all reach a shape, with the leaves of the
        `topic_expand_max_roots` best Topics; a shape holding only rivals is marked as one.
        The availability probe reads each confirmed candidate once, under the part's
        deadline or the fail-open budget, so it reads at most `detect_k + topic_expand_max_roots *
        topic_expand_max_members` dcids plus the rivals."""
        config = settings()
        topics = [f"dc/topic/T{t}" for t in range(config.topic_expand_max_roots + 1)]
        leaves = {
            topic: [f"Leaf_{t}_{j}" for j in range(config.topic_expand_max_members)]
            for t, topic in enumerate(topics)
        }
        nodes: dict = {
            topic: {
                "arcs": {
                    "relevantVariable": {
                        "nodes": [
                            {"dcid": leaf, "types": ["StatisticalVariable"]}
                            for leaf in leaves[topic]
                        ]
                    }
                }
            }
            for topic in topics
        }
        for leaf in (leaf for found in leaves.values() for leaf in found):
            nodes[leaf] = {"label": "l", "arcs": _std_sv_arcs("Country", f"metric_{leaf}")}
        reals = [f"Real_{i}" for i in range(config.detect_k - len(topics) + rivals)]
        for i, dcid in enumerate(reals):
            nodes[dcid] = {"label": "r", "arcs": _std_sv_arcs("Country", f"metric_{i}")}
        sv_scores = {
            **dict.fromkeys(topics, 1.0),
            **{d: 0.99 - i * 0.001 for i, d in enumerate(reals)},
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
        probe = graph.observation_facets_batch
        probed: list[tuple[list[str], float | None]] = []

        def recording(stat_vars, entities, needs_dates=False, **kwargs):
            probed.append((list(stat_vars), bounded_deadline(kwargs.get("deadline"))))
            return probe(stat_vars, entities, needs_dates, **kwargs)

        graph.observation_facets_batch = recording
        rival_svs = frozenset(reals[len(reals) - rivals :])
        deadline = time.monotonic() + 60
        started = time.monotonic()

        shapes = derive_shapes(
            confirmed_svs=[*topics, *reals],
            graph=graph,
            sv_scores=sv_scores,
            sv_types=dict.fromkeys(topics, "Topic"),
            rival_svs=rival_svs,
            place_dcids=["country/KEN"],
            deadline=deadline,
        )

        expanded = [
            leaf for topic in topics[: config.topic_expand_max_roots] for leaf in leaves[topic]
        ]
        assert {sv for s in shapes if s.rival for sv in (s.sv_arc_facts or {})} == rival_svs
        assert {sv for s in shapes for sv in (s.sv_arc_facts or {})} == {*reals, *expanded}
        ((stat_vars, probe_deadline),) = probed
        assert sorted(stat_vars) == sorted([*reals, *expanded])
        assert len(stat_vars) <= (
            config.detect_k
            + config.topic_expand_max_roots * config.topic_expand_max_members
            + rivals
        )
        # The probe fails open, so it reads under the fail-open budget or the part's deadline,
        # whichever ends first.
        assert_fail_open_deadline(probe_deadline, requested=deadline, started=started)


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
