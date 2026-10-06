"""Tests for interpret.recall — SV detection and entity resolution."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.interpret import recall
from tests.fixtures import FakeGraph, make_resolver


def run(coro):
    return asyncio.run(coro)


class TestRecall:
    def test_candidates_carry_score_type_of_sentence(self):
        """Candidates retain dcid, score, type, and sentence; legacy lists derive from them."""
        graph = FakeGraph()
        result = run(
            recall(
                "government education spending",
                ["Kenya"],
                graph=graph,
                resolver=make_resolver(graph),
                raw_query="government education spending Kenya",
            )
        )
        assert result.candidates, "expected at least one recalled candidate"
        for c in result.candidates:
            assert isinstance(c.dcid, str) and c.dcid
            assert isinstance(c.score, float)
            # type_of/sentence are optional (None allowed) on this fixture entry.
        assert [c.dcid for c in result.candidates] == result.candidate_svs
        assert [c.score for c in result.candidates] == result.candidate_sv_scores

    def test_recall_preserves_detect_order(self):
        """Recall preserves score order because `derive_shapes` trims the list before confirming."""
        graph = FakeGraph(
            nodes={},
            obs={},
            resolve={},
            detect={
                "multi score query": {
                    "svs": ["sv/Low", "sv/High", "sv/Mid"],
                    "cosine_scores": [0.61, 0.91, 0.75],
                }
            },
        )
        result = run(recall("multi score query", [], graph=graph, resolver=make_resolver(graph)))
        assert [c.dcid for c in result.candidates] == ["sv/High", "sv/Mid", "sv/Low"]
        assert result.candidate_sv_scores == [0.91, 0.75, 0.61]

    def test_entity_resolution_ethiopia(self):
        graph = FakeGraph()
        result = run(
            recall(
                "health ODA grants to Ethiopia",
                ["Ethiopia"],
                graph=graph,
                resolver=make_resolver(graph),
            )
        )
        assert result.resolved_entity_names.get("Ethiopia") == "country/ETH"

    def test_entity_resolution_usa(self):
        graph = FakeGraph()
        result = run(
            recall(
                "health ODA grants from USA to Kenya",
                ["USA", "Kenya"],
                graph=graph,
                resolver=make_resolver(graph),
            )
        )
        assert result.resolved_entity_names.get("USA") == "country/USA"
        assert result.resolved_entity_names.get("Kenya") == "country/KEN"

    def test_unknown_entity_not_in_resolved(self):
        graph = FakeGraph()
        result = run(
            recall("ODA to Atlantis", ["Atlantis"], graph=graph, resolver=make_resolver(graph))
        )
        assert "Atlantis" not in result.resolved_entity_names

    def test_no_entities(self):
        graph = FakeGraph()
        result = run(
            recall("health ODA grants to Kenya", [], graph=graph, resolver=make_resolver(graph))
        )
        assert result.resolved_entity_names == {}

    def test_graph_error_propagates(self):
        from search.engine.errors import GraphInfraError

        graph = FakeGraph(raise_on_call=True)
        with pytest.raises(GraphInfraError):
            run(
                recall(
                    "health ODA grants", ["Ethiopia"], graph=graph, resolver=make_resolver(graph)
                )
            )

    def test_graph_error_propagates_two_entities(self):
        """Plain gather must still raise loud with two entities (detect + 2 resolves)."""
        from search.engine.errors import GraphInfraError

        graph = FakeGraph(raise_on_call=True)
        with pytest.raises(GraphInfraError):
            run(
                recall(
                    "health ODA grants",
                    ["Ethiopia", "Kenya"],
                    graph=graph,
                    resolver=make_resolver(graph),
                )
            )


class _CountingGraph(FakeGraph):
    """FakeGraph wrapper that counts detect_svs calls, for call-count assertions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detect_calls: list[str] = []

    def detect_svs(self, query: str):
        self.detect_calls.append(query)
        return super().detect_svs(query)


class TestAcronymCanonicalization:
    """GNI, U5MR, IMR, and PPG expand only for bare-variable detection; other fields stay unchanged.
    GDP is excluded."""

    def test_bare_acronym_query_is_expanded(self):
        graph = _CountingGraph(
            nodes={},
            obs={},
            resolve={},
            detect={"gross national income": {"svs": ["sv/GNI"], "cosine_scores": [0.82]}},
        )
        result = run(recall("GNI", [], graph=graph, resolver=make_resolver(graph), raw_query="GNI"))
        assert graph.detect_calls == ["gross national income"]
        assert result.candidate_svs == ["sv/GNI"]

    def test_acronym_within_a_longer_phrase_keeps_the_rest(self):
        """A qualifier phrase like "GNI per capita" must keep "per capita" --
        only the acronym token itself is replaced."""
        graph = _CountingGraph(
            nodes={},
            obs={},
            resolve={},
            detect={"gross national income per capita": {"svs": ["sv/X"], "cosine_scores": [0.9]}},
        )
        run(
            recall(
                "GNI per capita",
                [],
                graph=graph,
                resolver=make_resolver(graph),
                raw_query="GNI per capita",
            )
        )
        assert graph.detect_calls == ["gross national income per capita"]

    def test_case_insensitive(self):
        graph = _CountingGraph(
            nodes={},
            obs={},
            resolve={},
            detect={"carbon dioxide emissions": {"svs": ["sv/CO2"], "cosine_scores": [0.87]}},
        )
        run(recall("co2", [], graph=graph, resolver=make_resolver(graph), raw_query="co2"))
        assert graph.detect_calls == ["carbon dioxide emissions"]

    @pytest.mark.parametrize(
        ("variable_text", "expected_query"),
        [
            pytest.param(
                "debt service for PPG debt",
                "debt service for public and publicly guaranteed debt",
                id="ppg_expanded_in_phrase",
            ),
            pytest.param("mppgx stock", "mppgx stock", id="ppg_inside_a_word_unchanged"),
        ],
    )
    def test_ppg_expands_at_word_boundaries_only(self, variable_text, expected_query):
        graph = _CountingGraph()
        run(
            recall(
                variable_text,
                [],
                graph=graph,
                resolver=make_resolver(graph),
                raw_query=variable_text,
            )
        )
        assert graph.detect_calls == [expected_query]

    def test_unmapped_term_passes_through_unchanged(self):
        graph = _CountingGraph()
        run(
            recall(
                "health ODA grants",
                ["Ethiopia"],
                graph=graph,
                resolver=make_resolver(graph),
                raw_query="health ODA grants",
            )
        )
        assert graph.detect_calls == ["health ODA grants"]

    def test_disabled_via_config_passes_query_through_unchanged(self, search_settings):
        search_settings(acronym_canonicalize=False)
        graph = _CountingGraph(
            nodes={},
            obs={},
            resolve={},
            detect={"GNI": {"svs": ["sv/A"], "cosine_scores": [0.6]}},
        )
        run(recall("GNI", [], graph=graph, resolver=make_resolver(graph), raw_query="GNI"))
        assert graph.detect_calls == ["GNI"]


class TestRelevanceFloor:
    """The 0.30 default sits 0.27 points below the 0.5738 minimum across 57 goldens.

    Topic candidates bypass the floor because they seed expansion from the recall pool."""

    @pytest.mark.parametrize(
        "floor",
        [
            pytest.param(None, id="default_floor"),
            pytest.param(0.3, id="explicit_floor_030"),
        ],
    )
    def test_floor_drops_candidates_below_it(self, search_settings, floor):
        """The default and explicit 0.30 floor drop the same weak candidate."""
        if floor is not None:
            search_settings(recall_score_floor=floor)
        graph = FakeGraph(
            nodes={},
            obs={},
            resolve={},
            detect={"q": {"svs": ["sv/Strong", "sv/Weak"], "cosine_scores": [0.9, 0.1]}},
        )
        result = run(recall("q", [], graph=graph, resolver=make_resolver(graph), raw_query="q"))
        assert result.candidate_svs == ["sv/Strong"]

    def test_floor_set_to_zero_is_a_no_op(self, search_settings):
        search_settings(recall_score_floor=0.0)
        graph = FakeGraph(
            nodes={},
            obs={},
            resolve={},
            detect={"q": {"svs": ["sv/Weak"], "cosine_scores": [0.01]}},
        )
        result = run(recall("q", [], graph=graph, resolver=make_resolver(graph), raw_query="q"))
        assert result.candidate_svs == ["sv/Weak"]

    @pytest.mark.parametrize(
        "types",
        [
            pytest.param({"dc/topic/Weak": "Topic"}, id="authoritative_topic_type"),
            pytest.param(None, id="missing_type_dcid_fallback"),
        ],
    )
    def test_topic_candidate_survives_floor(self, search_settings, types):
        """Topic classification and the dcid fallback retain the low-score root."""
        search_settings(recall_score_floor=0.3)
        graph = FakeGraph(
            nodes={},
            obs={},
            resolve={},
            detect={
                "q": {
                    "svs": ["dc/topic/Weak", "sv/Weak"],
                    "cosine_scores": [0.1, 0.1],
                    **({"types": types} if types is not None else {}),
                }
            },
        )
        result = run(recall("q", [], graph=graph, resolver=make_resolver(graph), raw_query="q"))
        assert result.candidate_svs == ["dc/topic/Weak"]

    def test_floor_can_empty_the_pool_entirely(self, search_settings):
        search_settings(recall_score_floor=0.99)
        graph = FakeGraph(
            nodes={},
            obs={},
            resolve={},
            detect={"q": {"svs": ["sv/A"], "cosine_scores": [0.5]}},
        )
        result = run(recall("q", [], graph=graph, resolver=make_resolver(graph), raw_query="q"))
        assert result.candidate_svs == []
