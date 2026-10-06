"""Tests for the computed near-neighbour margin check, which complements the
binder's `also_plausible` signal and catches errors such as literacy queries
retrieving fertility data. A 0.0046 forest-area margin is confusable; a 0.243
fertility-rate margin is clearly dominant.

Unit tests cover token similarity, shape distinction, and exemptions. Pipeline
tests pin warnings, candidate status, and ordering with synthetic and recorded data.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.config import settings
from search.engine.region import (
    NEAR_NEIGHBOR_AMBIGUOUS,
    RETRIEVAL_SCORE_WEAK,
    RegionResult,
    _best_near_neighbor_index,
    _discriminating_token,
    _label_token_similarity,
    _near_neighbor_check,
    _representative_score_is_weak,
    _shapes_semantically_distinct,
    resolve_variable,
)
from search.engine.shape import ShapeDraft
from tests.fixtures import FakeGraph, make_resolver


def _shape(
    shape_id: str,
    *,
    pop_type: str = "Person",
    meas_prop: str = "count",
    stat_type: str = "measuredValue",
    meas_qual: str | None = None,
    score: float = 0.8,
    label: str | None = None,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=label if label is not None else shape_id,
        pop_type_dcid=pop_type,
        meas_prop_dcid=meas_prop,
        stat_type_dcid=stat_type,
        meas_qual_dcid=meas_qual,
        meas_denom_dcid=None,
        slot_keys=(),
        representative_score=score,
    )


# ---------------------------------------------------------------------------
# _label_token_similarity: pure token-set Jaccard
# ---------------------------------------------------------------------------


def test_label_token_similarity_identical_tokens():
    assert _label_token_similarity("forest area", "Forest Area") == 1.0


def test_label_token_similarity_disjoint_tokens():
    assert _label_token_similarity("forest area", "population count") == 0.0


def test_label_token_similarity_partial_overlap():
    # The token sets share one of four unique terms, so their Jaccard score is 1/4.
    assert _label_token_similarity("gdp per capita", "gdp nominal") == 0.25


def test_label_token_similarity_empty_input_is_zero():
    assert _label_token_similarity("", "forest area") == 0.0
    assert _label_token_similarity("forest area", "") == 0.0


def test_label_token_similarity_strips_parenthetical_punctuation():
    """Parenthetical text tokenizes like plain text."""
    assert _label_token_similarity("... (per person)", "... per person") == 1.0


# ---------------------------------------------------------------------------
# _discriminating_token: punctuation-robust tokenization (gov-health-pc-kenya bug)
# ---------------------------------------------------------------------------


def test_discriminating_token_finds_discriminator_across_parenthetical_label():
    """Parentheses keep `(per person)` token-compatible with `per person`, so the query's
    "per" is shared and "government" still surfaces as the discriminator."""
    token = _discriminating_token(
        query="government health expenditure per capita",
        chosen_label="Domestic general government current health expenditure (per person)",
        neighbor_label="Healthcare expenditure per person",
    )
    assert token == "government"


def test_discriminating_token_still_bails_out_on_genuine_ambiguity():
    """Unique words from both labels leave the query ambiguous."""
    token = _discriminating_token(
        query="adult fertility literacy rate",
        chosen_label="Adult fertility rate",
        neighbor_label="Literacy rate",
    )
    assert token is None


# ---------------------------------------------------------------------------
# _shapes_semantically_distinct: five-tuple concept comparison
# ---------------------------------------------------------------------------


def test_semantically_distinct_true_when_meas_prop_differs():
    a = _shape("a", meas_prop="area")
    b = _shape("b", meas_prop="count")
    assert _shapes_semantically_distinct(a, b) is True


def test_semantically_distinct_true_when_pop_type_differs():
    a = _shape("a", pop_type="Forest")
    b = _shape("b", pop_type="Person")
    assert _shapes_semantically_distinct(a, b) is True


def test_semantically_distinct_false_for_same_concept_variant():
    """Nominal and Real GDP share `meas_prop` and `pop_type`; their qualifier
    difference makes them variants of one concept.
    """
    nominal = _shape(
        "gdp_nominal", pop_type="EconomicActivity", meas_prop="amount", meas_qual="Nominal"
    )
    real = _shape("gdp_real", pop_type="EconomicActivity", meas_prop="amount", meas_qual="Real")
    assert _shapes_semantically_distinct(nominal, real) is False


# ---------------------------------------------------------------------------
# _best_near_neighbor_index
# ---------------------------------------------------------------------------


def test_best_near_neighbor_index_picks_highest_scoring_distinct_candidate():
    chosen = _shape("chosen", meas_prop="area", score=0.9)
    # Pre-sorted score-descending, matching resolve_variable's ranked_shapes/capped_shapes.
    capped = [
        chosen,
        _shape("high", meas_prop="count", score=0.7),
        _shape("low", meas_prop="count", score=0.5),
    ]
    assert _best_near_neighbor_index(chosen, 0, capped) == 1


def test_best_near_neighbor_index_skips_chosen_own_slot():
    chosen = _shape("chosen", meas_prop="area", score=0.9)
    capped = [chosen]
    assert _best_near_neighbor_index(chosen, 0, capped) is None


def test_best_near_neighbor_index_none_when_only_same_concept_variants():
    chosen = _shape(
        "chosen", pop_type="EconomicActivity", meas_prop="amount", meas_qual="Nominal", score=0.9
    )
    capped = [
        chosen,
        _shape(
            "variant", pop_type="EconomicActivity", meas_prop="amount", meas_qual="Real", score=0.89
        ),
    ]
    assert _best_near_neighbor_index(chosen, 0, capped) is None


# ---------------------------------------------------------------------------
# _near_neighbor_check: entry-condition exemptions + margin/similarity computation
# ---------------------------------------------------------------------------


def test_near_neighbor_check_none_for_an_unscored_chosen_shape():
    """An unscored shape has no margin to compare. A score of 1.0 is valid."""
    unscored = _shape("chosen", meas_prop="area", score=None)
    capped = [unscored, _shape("other", meas_prop="count", score=0.5)]
    assert _near_neighbor_check("variable text", unscored, 0, capped) is None


def test_near_neighbor_check_compares_a_shape_scored_exactly_one():
    chosen = _shape("chosen", meas_prop="area", score=1.0)
    other = _shape("other", meas_prop="count", score=0.5)
    result = _near_neighbor_check("variable text", chosen, 0, [chosen, other])
    assert result is not None
    neighbor_index, margin, _similarity = result
    assert neighbor_index == 1
    assert margin == pytest.approx(0.5)


def test_near_neighbor_check_none_when_no_distinct_candidate():
    chosen = _shape("chosen", meas_prop="area", score=0.9)
    capped = [chosen]
    assert _near_neighbor_check("variable text", chosen, 0, capped) is None


def test_near_neighbor_check_returns_margin_and_similarity():
    chosen = _shape("chosen", meas_prop="area", score=0.8230, label="Area Forest")
    other = _shape("other", meas_prop="count", score=0.8184, label="Count Person")
    capped = [chosen, other]
    result = _near_neighbor_check("forest area", chosen, 0, capped)
    assert result is not None
    neighbor_index, margin, similarity = result
    assert neighbor_index == 1
    assert margin == pytest.approx(0.8230 - 0.8184)
    assert similarity == _label_token_similarity("forest area", "Area Forest")


# ---------------------------------------------------------------------------
# _representative_score_is_weak: the shared RETRIEVAL_SCORE_WEAK-band predicate
# ---------------------------------------------------------------------------


def test_representative_score_is_weak_true_inside_band():
    midpoint = (settings().relevance_threshold + settings().weak_score_threshold) / 2
    assert _representative_score_is_weak(midpoint) is True


def test_representative_score_is_weak_false_at_or_below_relevance_floor():
    assert _representative_score_is_weak(settings().relevance_threshold) is False
    assert _representative_score_is_weak(settings().relevance_threshold - 0.01) is False


def test_representative_score_is_weak_false_at_or_above_weak_threshold():
    assert _representative_score_is_weak(settings().weak_score_threshold) is False
    assert _representative_score_is_weak(settings().weak_score_threshold + 0.01) is False


def test_representative_score_is_weak_false_for_a_perfect_score():
    """1.0 sits above settings().weak_score_threshold, so it is no more weak than
    any other confident-match score."""
    assert _representative_score_is_weak(1.0) is False


# A weak chosen score suppresses margin comparisons, even against a strong neighbour.


def test_near_neighbor_check_none_when_chosen_score_weak_despite_thin_margin():
    """A 0.01 margin is below the 0.025 default but carries little signal when
    the chosen score is weak, so the check returns no comparison.
    """
    weak_score = (settings().relevance_threshold + settings().weak_score_threshold) / 2
    chosen = _shape("chosen", meas_prop="area", score=weak_score)
    neighbor = _shape("neighbor", meas_prop="count", score=weak_score - 0.01)
    assert _near_neighbor_check("variable text", chosen, 0, [chosen, neighbor]) is None


def test_near_neighbor_check_weak_exemption_blind_to_population_type():
    """The weak-score exemption applies when the neighbour shares the
    chosen shape's population type.
    """
    weak_score = (settings().relevance_threshold + settings().weak_score_threshold) / 2
    chosen = _shape("chosen", pop_type="Person", meas_prop="fertilityRate", score=weak_score)
    neighbor = _shape(
        "neighbor", pop_type="Person", meas_prop="literacyRate", score=weak_score - 0.01
    )
    assert _near_neighbor_check("variable text", chosen, 0, [chosen, neighbor]) is None


def test_near_neighbor_check_still_fires_when_chosen_score_is_strong():
    """A strong chosen score triggers the margin check."""
    chosen = _shape("chosen", meas_prop="area", score=0.90)
    neighbor = _shape("neighbor", meas_prop="count", score=0.89)
    assert _near_neighbor_check("variable text", chosen, 0, [chosen, neighbor]) is not None


# Pipeline tests use a fixed binder and a confusable forest-area pair with a 0.0046 margin.
_SV_FOREST = "Area_Forest"
_SV_POPULATION = "Count_Person"

_NODES: dict = {
    "country/BRA": {"label": "Brazil", "type": "Country"},
    _SV_FOREST: {
        "label": "Forest Area",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Forest Area"}]},
            "populationType": {"nodes": [{"dcid": "Forest"}]},
            "measuredProperty": {"nodes": [{"dcid": "area"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _SV_POPULATION: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Count Person"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}

# GDP Nominal/Real: same concept (meas_prop=amount, pop_type=EconomicActivity),
# differing only in measurementQualifier, a same-concept five-tuple variant.
_SV_GDP_NOMINAL = "Amount_EconomicActivity_GDP_Nominal"
_SV_GDP_REAL = "Amount_EconomicActivity_GDP_Real"

_NODES_GDP: dict = {
    "country/BRA": {"label": "Brazil", "type": "Country"},
    _SV_GDP_NOMINAL: {
        "label": "GDP Nominal",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "GDP Nominal"}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        },
    },
    _SV_GDP_REAL: {
        "label": "GDP Real",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "GDP Real"}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementQualifier": {"nodes": [{"dcid": "Real"}]},
        },
    },
}

_RESOLVE: dict = {"Brazil": "country/BRA"}


class _FixedBindLLM:
    """Elects capped_shapes[0] unconditionally, with a caller-supplied also_plausible."""

    def __init__(self, also_plausible: list[int] | None = None):
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=self._also_plausible,
            ask=None,
        ), None


def _run(
    *,
    variable: str,
    nodes: dict,
    scores: list[float],
    svs: list[str],
    also_plausible: list[int] | None = None,
    entity: str | None = "Brazil",
) -> RegionResult:
    detect_query = variable
    detect = {detect_query: {"svs": svs, "cosine_scores": scores}}
    _observation_entity = _RESOLVE[entity] if entity is not None else "country/BRA"
    obs = {
        f"{sv}|{_observation_entity}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 10}
        ]
        for sv in svs
    }
    graph = FakeGraph(
        nodes=nodes,
        detect=detect,
        resolve=_RESOLVE,
        obs=obs,
        children={"Earth": {"Country": [_observation_entity]}} if entity is None else {},
    )
    return asyncio.run(
        resolve_variable(
            variable,
            entities=[] if entity is None else [entity],
            date_request=None,
            detect_query=detect_query,
            role_query=variable if entity is None else f"{variable} in {entity}",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(also_plausible=also_plausible),
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
        )
    )


class TestGateFiresDowngradesToCandidates:
    """Thin margin (0.0046) + a query that names neither shape's real concept
    ("Forest Area" / "Count Person") -> low query/label overlap -> candidates,
    both shapes, NEAR_NEIGHBOR_AMBIGUOUS present."""

    def _result(self) -> RegionResult:
        return _run(
            variable="land use statistic",
            nodes=_NODES,
            svs=[_SV_FOREST, _SV_POPULATION],
            scores=[0.8230, 0.8184],
        )

    def test_status_is_candidates(self):
        assert self._result().status == "candidates"

    def test_both_shapes_present(self):
        result = self._result()
        spec_ids = {s.spec_id for s in result.specs}
        assert len(result.specs) == 2
        assert len(spec_ids) == 2

    def test_warning_present(self):
        result = self._result()
        assert any(w.code == NEAR_NEIGHBOR_AMBIGUOUS for w in result.warnings)


def test_place_less_candidates_keep_availability_unknown_but_grade_relevance():
    """Candidate scores grade relevance without a place probe. The chosen
    candidate clears the weak-score threshold; the alternate falls below it.
    """
    result = _run(
        variable="generic statistic",
        nodes=_NODES,
        svs=[_SV_FOREST, _SV_POPULATION],
        scores=[0.80, 0.70],
        also_plausible=[1],
        entity=None,
    )
    assert result.status == "candidates"
    assert result.candidate_near_neighbor_spec_ids == frozenset()
    assert result.candidate_scores is not None
    by_stat_var = {spec.resolved.members[0].ref.dcid: spec for spec in result.specs}
    assert all(spec.resolved.available_at_place is None for spec in by_stat_var.values())
    for spec in by_stat_var.values():
        expected = (
            "high"
            if result.candidate_scores[spec.spec_id] >= settings().weak_score_threshold
            else "medium"
        )
        assert spec.resolved.confidence == expected


def test_wide_margin_stays_definite():
    """Both shapes survive the unknown-score floor, but a wide margin (0.20, well
    above SEARCH_NEAR_MARGIN's default 0.025) means no downgrade."""
    result = _run(
        variable="land use statistic",
        nodes=_NODES,
        svs=[_SV_FOREST, _SV_POPULATION],
        scores=[0.90, 0.70],
    )
    assert result.status == "definite"
    assert not any(w.code == NEAR_NEIGHBOR_AMBIGUOUS for w in result.warnings)


def test_same_concept_variant_stays_definite():
    """GDP Nominal and Real share a concept, so their 0.002 margin leaves the
    gate quiet.
    """
    result = _run(
        variable="gdp",
        nodes=_NODES_GDP,
        svs=[_SV_GDP_NOMINAL, _SV_GDP_REAL],
        scores=[0.900, 0.898],
    )
    assert result.status == "definite"
    assert not any(w.code == NEAR_NEIGHBOR_AMBIGUOUS for w in result.warnings)


def test_weak_chosen_score_stays_definite_despite_thin_margin():
    """A 0.01 margin is below the 0.025 default, but both scores are weak.
    The answer stays definite with `RETRIEVAL_SCORE_WEAK` and no ambiguity warning.
    """
    weak_score = (settings().relevance_threshold + settings().weak_score_threshold) / 2
    result = _run(
        variable="land use statistic",
        nodes=_NODES,
        svs=[_SV_FOREST, _SV_POPULATION],
        scores=[weak_score, weak_score - 0.01],
    )
    assert result.status == "definite"
    codes = [w.code for w in result.warnings]
    assert NEAR_NEIGHBOR_AMBIGUOUS not in codes
    assert RETRIEVAL_SCORE_WEAK in codes


class TestHighLabelSimilarityStaysDefinite:
    """A 0.0046 margin leaves the gate quiet when the query names the chosen
    label, "Forest Area".
    """

    def _result(self) -> RegionResult:
        return _run(
            variable="forest area",
            nodes=_NODES,
            svs=[_SV_FOREST, _SV_POPULATION],
            scores=[0.8230, 0.8184],
        )

    def test_status_is_definite(self):
        assert self._result().status == "definite"

    def test_no_warning(self):
        result = self._result()
        assert not any(w.code == NEAR_NEIGHBOR_AMBIGUOUS for w in result.warnings)


class TestAlsoPlausibleAndGateComposing:
    """The margin gate agrees with `also_plausible` without duplicating specs."""

    def _result(self) -> RegionResult:
        return _run(
            variable="land use statistic",
            nodes=_NODES,
            svs=[_SV_FOREST, _SV_POPULATION],
            scores=[0.8230, 0.8184],
            also_plausible=[1],
        )

    def test_status_is_candidates(self):
        assert self._result().status == "candidates"

    def test_no_duplicate_specs(self):
        result = self._result()
        spec_ids = [s.spec_id for s in result.specs]
        assert len(spec_ids) == 2
        assert len(set(spec_ids)) == 2

    def test_warning_still_present(self):
        result = self._result()
        assert any(w.code == NEAR_NEIGHBOR_AMBIGUOUS for w in result.warnings)


def test_single_candidate_shape_unaffected():
    """With one surviving shape, the gate has no neighbour to compare."""
    result = _run(
        variable="forest area",
        nodes=_NODES,
        svs=[_SV_FOREST],
        scores=[0.90],
    )
    assert result.status == "definite"
    assert not any(w.code == NEAR_NEIGHBOR_AMBIGUOUS for w in result.warnings)


# ---------------------------------------------------------------------------
# Equal retrieval scores prefer the shape with more members.
# This tie-break determines which candidate the near-neighbour check selects.
# ---------------------------------------------------------------------------

_SV_CHOSEN_TIEBREAK = "Chosen_Tiebreak_Metric"
_SV_SOLO = "Solo_Metric"
_SV_TRIO_A = "Trio_Metric_A"
_SV_TRIO_B = "Trio_Metric_B"
_SV_TRIO_C = "Trio_Metric_C"


def _tiebreak_node(sv: str, pop_type: str, meas_prop: str, label: str) -> dict:
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": pop_type}]},
            "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    }


_TIEBREAK_NODES: dict = {
    "country/BRA": {"label": "Brazil", "type": "Country"},
    _SV_CHOSEN_TIEBREAK: _tiebreak_node(
        _SV_CHOSEN_TIEBREAK, "ThingChosen", "propChosenTiebreak", "Chosen Tiebreak Metric"
    ),
    # Solo: a single SV -> its own shape group, breadth 1.
    _SV_SOLO: _tiebreak_node(_SV_SOLO, "ThingSolo", "propSolo", "Solo Metric"),
    # Trio: three SVs sharing one five-tuple (populationType=ThingTrio,
    # measuredProperty=propTrio) -> derive_shapes groups them into ONE
    # ShapeDraft with breadth 3 (see retrieval/derive.derive_shapes'
    # "Group confirmed SVs by five-tuple" step).
    _SV_TRIO_A: _tiebreak_node(_SV_TRIO_A, "ThingTrio", "propTrio", "Trio Metric"),
    _SV_TRIO_B: _tiebreak_node(_SV_TRIO_B, "ThingTrio", "propTrio", "Trio Metric"),
    _SV_TRIO_C: _tiebreak_node(_SV_TRIO_C, "ThingTrio", "propTrio", "Trio Metric"),
}


def test_ranked_shapes_breadth_tiebreak_prefers_broader_shape_at_equal_score():
    """At equal scores, Trio's 3 members beat Solo's 1. The selected neighbour
    must belong to Trio.
    """
    result = _run(
        variable="generic statistic",
        nodes=_TIEBREAK_NODES,
        svs=[_SV_CHOSEN_TIEBREAK, _SV_TRIO_A, _SV_TRIO_B, _SV_TRIO_C, _SV_SOLO],
        scores=[0.830, 0.825, 0.825, 0.825, 0.825],
    )
    codes = [w.code for w in result.warnings]
    assert NEAR_NEIGHBOR_AMBIGUOUS in codes
    fired_svs = {
        sv.ref.dcid
        for spec in result.specs
        if spec.spec_id in result.candidate_near_neighbor_spec_ids
        for sv in spec.resolved.members
    }
    assert fired_svs & {_SV_TRIO_A, _SV_TRIO_B, _SV_TRIO_C}
    assert _SV_SOLO not in fired_svs


# ---------------------------------------------------------------------------
# The computed neighbour must survive both the pipeline's alternate budget
# and the response's candidate cap, even when the model omits it.
# ---------------------------------------------------------------------------

from search.engine.assemble import build_candidates_part  # noqa: E402


def _filler_node(index: int, meas_prop: str) -> dict:
    label = f"Filler {index} Statistic"
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": "Thing"}]},
            "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    }


_SV_CHOSEN = "Chosen_Metric"
_SV_NEIGHBOR = "Neighbor_Metric"


def _neighbor_scenario_nodes(filler_count: int) -> tuple[dict, list[str], list[float]]:
    """Build (nodes, svs, scores) for chosen (index 0) + neighbor (index 1,
    margin 0.002 inside SEARCH_NEAR_MARGIN) + `filler_count` additional
    distinct-concept shapes at descending scores, all above
    SEARCH_UNKNOWN_SCORE_THRESHOLD. Scores strictly descending -> capped_shapes
    order matches svs order exactly (region/pipeline.py's ranked_shapes sort)."""
    nodes = {
        "country/BRA": {"label": "Brazil", "type": "Country"},
        _SV_CHOSEN: {
            "label": "Chosen Metric",
            "type": "StatisticalVariable",
            "arcs": {
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "name": {"nodes": [{"value": "Chosen Metric"}]},
                "populationType": {"nodes": [{"dcid": "Thing"}]},
                "measuredProperty": {"nodes": [{"dcid": "propChosen"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
            },
        },
        _SV_NEIGHBOR: {
            "label": "Neighbor Metric",
            "type": "StatisticalVariable",
            "arcs": {
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "name": {"nodes": [{"value": "Neighbor Metric"}]},
                "populationType": {"nodes": [{"dcid": "Thing"}]},
                "measuredProperty": {"nodes": [{"dcid": "propNeighbor"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
            },
        },
    }
    svs = [_SV_CHOSEN, _SV_NEIGHBOR]
    scores = [0.830, 0.828]
    for i in range(2, 2 + filler_count):
        sv = f"Filler_{i}_Metric"
        nodes[sv] = _filler_node(i, f"propFiller{i}")
        svs.append(sv)
        scores.append(0.80 - 0.01 * (i - 2))
    return nodes, svs, scores


def test_fired_neighbor_survives_five_or_more_also_plausible():
    """Five fillers at the default cap of 6 leave the computed neighbour in
    the final candidate set, even when the model omits it.
    """
    filler_count = settings().max_candidates - 1
    nodes, svs, scores = _neighbor_scenario_nodes(filler_count)
    # The model reports each filler and omits the computed neighbour at index 1.
    also_plausible = list(range(2, 2 + filler_count))
    result = _run(
        variable="generic statistic",
        nodes=nodes,
        svs=svs,
        scores=scores,
        also_plausible=also_plausible,
    )
    # At the default cap of 6, five alternates fit. The computed neighbour plus
    # five fillers exceed that budget, so Filler_6 is dropped and truncation is set.
    assert result.status == "candidates"
    sv_sets = [{sv.ref.dcid for sv in s.resolved.members} for s in result.specs]
    assert {_SV_CHOSEN} in sv_sets
    assert {_SV_NEIGHBOR} in sv_sets
    assert len(result.specs) == settings().max_candidates
    assert {"Filler_6_Metric"} not in sv_sets
    assert result.candidate_budget_truncated is True


def test_fired_neighbor_survives_response_max_candidates_floor():
    """With a response cap of 2 and 2 fillers, the pipeline budget keeps all
    four candidates. The response cap selects the elected shape and neighbour.
    """
    nodes, svs, scores = _neighbor_scenario_nodes(2)
    result = _run(
        variable="generic statistic",
        nodes=nodes,
        svs=svs,
        scores=scores,
        also_plausible=[2, 3],  # fillers only, neighbor (index 1) self-reported by nobody
    )
    assert result.status == "candidates"
    assert len(result.specs) == 4
    assert result.candidate_budget_truncated is False

    part = build_candidates_part(
        list(result.specs),
        variable_text="generic statistic",
        warnings=[],
        max_candidates=2,
        bind_ranked=result.near_neighbor_fired,
        budget_truncated=result.candidate_budget_truncated,
    )
    inner = part
    assert isinstance(inner, CandidatesPart)
    clamped_sv_sets = [{sv.ref.dcid for sv in s.resolved.members} for s in inner.candidates.specs]
    assert clamped_sv_sets == [{_SV_CHOSEN}, {_SV_NEIGHBOR}]
    assert inner.candidates.truncated is True


def test_fired_neighbor_wins_scarce_slot_over_models_own_top_alternate():
    """With a response cap of 2, the computed neighbour at index 1 outranks
    the model's filler choices at indices 3 and 7. The margin is 0.002.
    """
    nodes, svs, scores = _neighbor_scenario_nodes(6)
    result = _run(
        variable="generic statistic",
        nodes=nodes,
        svs=svs,
        scores=scores,
        also_plausible=[3, 7],  # The model prefers fillers 3 and 7 over neighbour 1.
    )
    assert result.status == "candidates"
    assert result.near_neighbor_fired is True
    assert len(result.specs) == 4
    assert result.candidate_budget_truncated is False

    part = build_candidates_part(
        list(result.specs),
        variable_text="generic statistic",
        warnings=[],
        max_candidates=2,
        bind_ranked=result.near_neighbor_fired,
        budget_truncated=result.candidate_budget_truncated,
    )
    inner = part
    assert isinstance(inner, CandidatesPart)
    clamped_sv_sets = [{sv.ref.dcid for sv in s.resolved.members} for s in inner.candidates.specs]
    assert clamped_sv_sets == [{_SV_CHOSEN}, {_SV_NEIGHBOR}]
    assert {"Filler_3_Metric"} not in clamped_sv_sets
    assert {"Filler_7_Metric"} not in clamped_sv_sets


def test_near_neighbor_signal_marks_only_elected_and_fired_candidates():
    """Only the elected shape and fired neighbour carry the gate signal."""
    nodes, svs, scores = _neighbor_scenario_nodes(2)
    result = _run(
        variable="generic statistic",
        nodes=nodes,
        svs=svs,
        scores=scores,
        also_plausible=[2, 3],
    )
    assert result.status == "candidates"
    by_stat_var = {spec.resolved.members[0].ref.dcid: spec for spec in result.specs}
    assert result.candidate_near_neighbor_spec_ids == frozenset(
        {
            by_stat_var[_SV_CHOSEN].spec_id,
            by_stat_var[_SV_NEIGHBOR].spec_id,
        }
    )
    assert by_stat_var[_SV_CHOSEN].resolved.confidence == "medium"
    assert by_stat_var[_SV_NEIGHBOR].resolved.confidence == "medium"
    assert by_stat_var["Filler_2_Metric"].resolved.confidence == "high"
    assert by_stat_var["Filler_3_Metric"].resolved.confidence == "high"


# ---------------------------------------------------------------------------
# Recorded end-to-end pairs cover a fired gate (nn-03), comfortable margins
# (nn-01, nn-02, nn-05), and the no-distinct-candidate exemption (nn-04).
# ---------------------------------------------------------------------------

from search.engine.region import phase_election  # noqa: E402
from search.models import CandidatesPart, NoDataPart, Resolution  # noqa: E402
from tests.engine._fake_coverage import empty_coverage_snapshot  # noqa: E402
from tests.engine._fake_sdmx import FakeSdmxClient  # noqa: E402
from tests.engine._harness import make_request, offline_resolve  # noqa: E402


class TestNn01MeaslesImmunizationCoverageFlagged:
    """nn-01 elects Urban measles coverage (0.8080) over Urban full
    immunization coverage (0.7185), a 0.0895 margin above 0.025. Kenya has no
    data for the elected StatVar, so the result carries a flagged match.
    """

    _QUERY = "measles immunization rate Kenya"

    def test_status_is_no_data(self):
        result = offline_resolve(make_request(self._QUERY))
        assert result.status == "no_data"

    def test_near_neighbor_warning_absent(self):
        result = offline_resolve(make_request(self._QUERY))
        codes = [w.code for w in result.parts[0].warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS not in codes

    def test_flagged_match_is_immunization_coverage(self):
        result = offline_resolve(make_request(self._QUERY))
        inner = result.parts[0]
        assert inner.status == "no_data"
        assert inner.no_data.reason == "no_observations"
        assert inner.no_data.flagged_match is not None
        sv_dcids = [sv.ref.dcid for sv in inner.no_data.flagged_match.resolved.members]
        assert sv_dcids, "expected the flagged match to resolve to at least one StatVar"
        assert all(ref == "WHO/WHS8_110" or ref.startswith("WHO/mslv_") for ref in sv_dcids), (
            f"expected the flagged match to resolve to a measles immunization coverage "
            f"StatVar, got {sv_dcids}"
        )


class TestNn02DptImmunizationStaysQuiet:
    """nn-02 compares DPT with HepB at a 0.0045 margin, so the gate fires with
    `RETRIEVAL_SCORE_WEAK`. Neither has Ethiopia data, so the result has no
    candidate specs and no ambiguity warning.
    """

    _QUERY = "DPT immunization coverage Ethiopia"

    def test_status_is_no_data(self):
        result = offline_resolve(make_request(self._QUERY))
        assert result.status == "no_data"

    def test_near_neighbor_warning_absent(self):
        result = offline_resolve(make_request(self._QUERY))
        codes = [w.code for w in result.parts[0].warnings]
        assert NEAR_NEIGHBOR_AMBIGUOUS not in codes
        assert "retrieval_score_weak" in codes


class TestNn03ChildMortalityFiresAtGateBoundary:
    """nn-03: 'child mortality rate Ethiopia'. Bind elects the under-14 death rate, and the
    under-4 mortality rate sits inside SEARCH_NEAR_MARGIN of it, so the gate fires. The
    elected rate has no Ethiopia data and the under-4 rate does, so the answer is no data
    with the under-4 rate as its alternate match, and the ambiguity warning, which needs two
    surviving readings, is retracted."""

    _QUERY = "child mortality rate Ethiopia"

    def test_the_gate_fires_on_the_under_4_rate(self, monkeypatch):
        checks: list[tuple[str, float]] = []
        real = phase_election._near_neighbor_check

        def _check(variable, shape_draft, chosen_index, capped_shapes):
            result = real(variable, shape_draft, chosen_index, capped_shapes)
            if result is not None:
                checks.append((capped_shapes[result[0]].label, result[1]))
            return result

        monkeypatch.setattr(phase_election, "_near_neighbor_check", _check)
        offline_resolve(make_request(self._QUERY))

        ((label, margin),) = checks
        assert label == "Mortality Rate of Children Younger Than 4 Years"
        assert margin < settings().near_margin

    def test_the_under_4_rate_is_the_alternate_match(self):
        result = offline_resolve(make_request(self._QUERY))
        assert isinstance(result, Resolution)
        inner = result.parts[0]
        assert isinstance(inner, NoDataPart)
        assert inner.no_data.gate == "only_alternate_has_data"
        assert inner.no_data.alternate_match is not None
        assert [sv.ref.dcid for sv in inner.no_data.alternate_match.resolved.members] == [
            "MortalityRate_Person_Upto4Years_AsFractionOf_Count_BirthEvent_LiveBirth"
        ]
        assert NEAR_NEIGHBOR_AMBIGUOUS not in [w.code for w in inner.warnings]
