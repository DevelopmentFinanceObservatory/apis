"""Tests for interpret.recall — SV detection and entity resolution."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future

import pytest

from code_index.shared import get_index
from entity_resolution import Resolved
from search.engine.clients.graph import DetectedSV, Detection, settled_un
from search.engine.errors import GraphInfraError
from search.engine.interpret import (
    Recall,
    _detect,
    _pooled,
    _states_a_word,
    _without_stated_words,
    recall,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import FakeCatalogue


def run(coro):
    return asyncio.run(coro)


def _resolved_dcids(result: Recall) -> dict[str, str]:
    """The dcid of each name recall resolved to one entity."""
    return {
        name: resolution.entity.dcid
        for name, resolution in result.resolutions.items()
        if isinstance(resolution, Resolved)
    }


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
        assert _resolved_dcids(result) == {"Ethiopia": "country/ETH"}

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
        assert _resolved_dcids(result) == {"USA": "country/USA", "Kenya": "country/KEN"}

    def test_unknown_entity_not_in_resolved(self):
        graph = FakeGraph()
        result = run(
            recall("ODA to Atlantis", ["Atlantis"], graph=graph, resolver=make_resolver(graph))
        )
        assert set(result.resolutions) == {"Atlantis"}
        assert _resolved_dcids(result) == {}

    def test_no_entities(self):
        graph = FakeGraph()
        result = run(
            recall("health ODA grants to Kenya", [], graph=graph, resolver=make_resolver(graph))
        )
        assert result.resolutions == {}

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


_CRS = "provenance/OECDCreditorReportingSystem"
_DAC1 = "provenance/OECDDAC1"
_DAC2A = "provenance/OECDDAC2A"
_IDS = "provenance/WorldBankInternationalDebtStatistics"


def _dcid_of(provenance: str, measured_property: str) -> str:
    """The first packaged index row of `provenance` with `measured_property`, StatVars the
    index lists included."""
    index = get_index()
    return next(
        row.dcid
        for row in (*index.statvars, *index.listed_statvars)
        if row.provenances == (provenance,) and row.measured_property == measured_property
    )


def _sv(dcid: str, score: float, *, rival: bool = False) -> DetectedSV:
    return DetectedSV(
        dcid=dcid, score=score, type_of="StatisticalVariable", sentence=None, rival=rival
    )


def _detection(
    candidates: list[DetectedSV],
    *,
    hits: list[DetectedSV] | None = None,
    un: Future | None = None,
) -> Detection:
    return Detection(
        candidates=tuple(candidates),
        un=settled_un(()) if un is None else un,
        hits=None if hits is None else tuple(hits),
    )


def _rows(svs: tuple[DetectedSV, ...] | None) -> list[tuple[str, float, bool]]:
    assert svs is not None
    return [(sv.dcid, sv.score, sv.rival) for sv in svs]


class TestWithoutStatedWords:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("bilateral ODA", ("ODA", {"bilateral"}), id="leading_word"),
            pytest.param(
                "Bilateral ODA to  Kenya",
                ("ODA to Kenya", {"bilateral"}),
                id="casefolded_and_whitespace_collapsed",
            ),
            pytest.param("ODA to Kenya", None, id="no_stated_word"),
            pytest.param("bilateral", None, id="nothing_left_without_it"),
            pytest.param("bilateral-only aid", None, id="hyphenated_token_is_another_word"),
            pytest.param("non-bilateral ODA", None, id="negated_token_is_another_word"),
        ],
    )
    def test_text_less_the_stated_words(self, text, expected):
        result = _without_stated_words(text, get_index())

        if expected is None:
            assert result is None
        else:
            assert result is not None
            rest, removed = result
            assert (rest, set(removed)) == expected


class TestStatesAWord:
    @pytest.mark.parametrize(
        ("provenance", "measured_property", "words", "expected"),
        [
            pytest.param(_CRS, "amount", {"bilateral"}, True, id="crs_amount"),
            pytest.param(_CRS, "grantEquivalent", {"bilateral"}, True, id="crs_grant_equivalent"),
            pytest.param(_DAC2A, "amount", {"bilateral"}, True, id="dac2a_amount"),
            pytest.param(_CRS, "amount", {"multilateral"}, False, id="another_word"),
            pytest.param(_DAC1, "amount", {"bilateral"}, False, id="dac1_codes_the_word"),
            pytest.param(_DAC2A, "attributedAmount", {"bilateral"}, False, id="dac2a_imputed_row"),
            pytest.param(
                _IDS, "amount", {"bilateral"}, False, id="listed_row_of_an_undeclared_table"
            ),
        ],
    )
    def test_a_row_joins_when_its_table_states_the_word_of_its_measured_property(
        self, provenance, measured_property, words, expected
    ):
        dcid = _dcid_of(provenance, measured_property)

        assert _states_a_word(dcid, words, get_index()) is expected

    def test_a_dcid_the_index_holds_no_row_for_states_nothing(self):
        assert _states_a_word("sv/NotInTheIndex", {"bilateral"}, get_index()) is False


class TestPooled:
    @pytest.mark.parametrize(
        ("first_score", "second_score"),
        [pytest.param(0.6, 0.9, id="second_higher"), pytest.param(0.9, 0.4, id="first_higher")],
    )
    def test_a_series_both_searches_return_appears_once_at_the_higher_score(
        self, first_score, second_score
    ):
        first = _detection([_sv("sv/A", first_score)])
        second = _detection([_sv("sv/A", second_score)])

        pooled = _pooled(first, second, lambda dcid: True)

        assert _rows(pooled.candidates) == [("sv/A", max(first_score, second_score), False)]

    def test_a_recalled_row_replaces_a_rival_family_row_of_the_same_series(self):
        first = _detection([_sv("sv/A", 0.8), _sv("sv/R", 0.7, rival=True)])
        second = _detection([_sv("sv/R", 0.4)])

        pooled = _pooled(first, second, lambda dcid: True)

        assert _rows(pooled.candidates) == [("sv/A", 0.8, False), ("sv/R", 0.4, False)]

    def test_only_the_second_series_the_predicate_admits_join(self):
        first = _detection([_sv("sv/A", 0.6), _sv("sv/B", 0.5), _sv("sv/C", 0.4)])
        second = _detection([_sv("sv/D", 0.7), _sv("sv/E", 0.9), _sv("sv/F", 0.8, rival=True)])

        pooled = _pooled(first, second, lambda dcid: dcid in {"sv/D", "sv/F"})

        assert _rows(pooled.candidates) == [
            ("sv/D", 0.7, False),
            ("sv/A", 0.6, False),
            ("sv/B", 0.5, False),
            ("sv/C", 0.4, False),
        ]

    def test_the_predicate_reads_the_dcid_a_series_stands_as(self, bind_catalogue):
        bind_catalogue(FakeCatalogue(successors={"sv/Old": "sv/New"}))
        seen: list[str] = []

        def joins(dcid: str) -> bool:
            seen.append(dcid)
            return True

        _pooled(_detection([]), _detection([_sv("sv/Old", 0.5)]), joins)

        assert seen == ["sv/New"]

    def test_the_first_searchs_un_leg_is_kept(self):
        un = settled_un((_sv("sv/Un", 0.5),))
        first = _detection([_sv("sv/A", 0.6)], un=un)
        second = _detection([_sv("sv/B", 0.9)])

        assert _pooled(first, second, lambda dcid: True).un is un

    @pytest.mark.parametrize(
        ("first_hits", "second_hits", "expected_hits"),
        [
            pytest.param(None, None, None, id="both_none"),
            pytest.param(
                [_sv("sv/A", 0.6), _sv("sv/R", 0.2, rival=True)],
                None,
                [("sv/B", 0.9, False), ("sv/A", 0.6, False), ("sv/R", 0.2, True)],
                id="only_the_first_has_hits",
            ),
            pytest.param(
                None,
                [_sv("sv/B", 0.9), _sv("sv/Z", 0.1)],
                [("sv/B", 0.9, False), ("sv/A", 0.6, False), ("sv/Z", 0.1, False)],
                id="only_the_second_has_hits",
            ),
        ],
    )
    def test_hits_read_as_candidates_where_a_detection_has_none(
        self, first_hits, second_hits, expected_hits
    ):
        first = _detection([_sv("sv/A", 0.6)], hits=first_hits)
        second = _detection([_sv("sv/B", 0.9)], hits=second_hits)

        pooled = _pooled(first, second, lambda dcid: True)

        if expected_hits is None:
            assert pooled.hits is None
        else:
            assert _rows(pooled.hits) == expected_hits


class _FailingGraph(_CountingGraph):
    """A counting graph whose `detect_svs` raises on one text."""

    def __init__(self, fails_on: str):
        super().__init__()
        self._fails_on = fails_on

    def detect_svs(self, query: str):
        detection = super().detect_svs(query)
        if query == self._fails_on:
            raise GraphInfraError(f"simulated transport error for {query!r}")
        return detection


class TestStatedWordSearch:
    @pytest.mark.parametrize(
        ("text", "searched"),
        [
            pytest.param("ODA to Kenya", ["ODA to Kenya"], id="no_stated_word"),
            pytest.param("bilateral", ["bilateral"], id="nothing_left_without_the_word"),
            pytest.param(
                "bilateral ODA to Kenya",
                ["ODA to Kenya", "bilateral ODA to Kenya"],
                id="stated_word",
            ),
        ],
    )
    def test_the_graph_is_searched_once_unless_the_text_holds_a_stated_word(self, text, searched):
        graph = _CountingGraph()

        run(recall(text, [], graph=graph, resolver=make_resolver(graph), raw_query=text))

        assert sorted(graph.detect_calls) == sorted(searched)

    def test_a_series_only_the_search_without_the_word_returns_joins_when_its_table_states_it(self):
        crs = _dcid_of(_CRS, "amount")
        dac1 = _dcid_of(_DAC1, "amount")
        ids = _dcid_of(_IDS, "amount")
        graph = _CountingGraph(
            nodes={},
            obs={},
            resolve={},
            detect={
                "bilateral ODA to Kenya": {"svs": [dac1], "cosine_scores": [0.8]},
                "ODA to Kenya": {"svs": [crs, ids], "cosine_scores": [0.7, 0.6]},
            },
        )

        result = run(
            recall(
                "bilateral ODA to Kenya",
                [],
                graph=graph,
                resolver=make_resolver(graph),
                raw_query="bilateral ODA to Kenya",
            )
        )

        recalled = {c.dcid for c in result.candidates if not c.rival}
        assert recalled >= {crs, dac1}
        assert ids not in recalled

    @pytest.mark.parametrize("fails_on", ["ODA to Kenya", "bilateral ODA to Kenya"])
    def test_a_failure_of_either_search_fails_the_request(self, fails_on):
        graph = _FailingGraph(fails_on)

        with pytest.raises(GraphInfraError):
            run(_detect(graph, "bilateral ODA to Kenya"))
