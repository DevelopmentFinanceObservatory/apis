"""Tests recall-score provenance in grouping, prompts, and confidence.

Topic-expanded members carry no measured score, including in mixed groups."""

from __future__ import annotations

import pytest

from search.engine.region.weak_score import _representative_score_is_weak
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.fixtures import FakeGraph

_TOPIC = "dc/topic/GDP"
_EXPANDED = "Expanded_Member_SV"
_RECALLED = "Recalled_SV"


def _graph() -> FakeGraph:
    """Builds one topic member and one scored recall candidate."""
    nodes: dict = {
        _TOPIC: {"arcs": {"relevantVariable": {"nodes": [{"dcid": _EXPANDED}]}}},
        _EXPANDED: {
            "label": "expanded member",
            "arcs": _std_sv_arcs("Country", "expandedMetric", "Expanded Metric"),
        },
        _RECALLED: {
            "label": "recalled",
            "arcs": _std_sv_arcs("Household", "recalledMetric", "Recalled Metric"),
        },
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _derive() -> list[ShapeDraft]:
    return derive_shapes(
        confirmed_svs=[_TOPIC, _RECALLED],
        graph=_graph(),
        sv_scores={_TOPIC: 0.90, _RECALLED: 0.5615},
        sv_types={_TOPIC: "Topic", _RECALLED: "StatisticalVariable"},
    )


def _by_member(shapes: list[ShapeDraft], member: str) -> ShapeDraft:
    return next(s for s in shapes if member in (s.sv_arc_facts or {}))


def test_expansion_only_group_has_no_representative_score():
    """Expansion members carry neither the Topic's 0.90 nor the sibling's 0.5615 score."""
    shapes = _derive()
    expanded = _by_member(shapes, _EXPANDED)
    assert expanded.recalled is False
    assert expanded.representative_score is None


def test_recalled_group_keeps_its_measured_score():
    """A recalled group keeps its measured cosine."""
    shapes = _derive()
    recalled = _by_member(shapes, _RECALLED)
    assert recalled.recalled is True
    assert recalled.representative_score == pytest.approx(0.5615)


def test_mixed_group_takes_its_scored_members_score():
    """A mixed group uses the recalled member's score."""
    graph = FakeGraph(
        nodes={
            _TOPIC: {"arcs": {"relevantVariable": {"nodes": [{"dcid": _EXPANDED}]}}},
            _EXPANDED: {
                "label": "expanded member",
                "arcs": _std_sv_arcs("Household", "sharedMetric"),
            },
            _RECALLED: {
                "label": "recalled",
                "arcs": _std_sv_arcs("Household", "sharedMetric", "Shared Metric"),
            },
        },
        obs={},
        detect={},
        resolve={},
    )
    shapes = derive_shapes(
        confirmed_svs=[_TOPIC, _RECALLED],
        graph=graph,
        sv_scores={_TOPIC: 0.90, _RECALLED: 0.5615},
        sv_types={_TOPIC: "Topic", _RECALLED: "StatisticalVariable"},
    )
    assert len(shapes) == 1
    shape = shapes[0]
    assert {_EXPANDED, _RECALLED} <= set(shape.sv_arc_facts or {})
    assert shape.representative_score == pytest.approx(0.5615)


def _shape(shape_id: str, *, meas_prop: str, score: float | None, label: str = "") -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=label or shape_id,
        pop_type_dcid="Country",
        meas_prop_dcid=meas_prop,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        representative_score=score,
    )


def test_representative_score_is_weak_false_for_an_unscored_shape():
    """An absent score yields ``False`` for weakness."""
    assert _representative_score_is_weak(None) is False


def _family_shape(shape_id: str, *, score: float | None) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts=None,
        representative_score=score,
    )


def _family_index():
    from code_index import CodeIndex, IndexStamp, StatVarRow

    def row(dcid: str, qualifier: str) -> StatVarRow:
        return StatVarRow(
            dcid=dcid,
            population_type="FinancialFlow",
            measured_property="amount",
            stat_type="measuredValue",
            measurement_qualifier=qualifier,
            measurement_denominator=None,
            observation_properties=(),
            constraint_properties={},
        )

    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test-hash",
        ),
        declarations=(),
        codes=(),
        statvars=[
            row("SV_FF_amount_Nominal", "Nominal"),
            row("SV_FF_amount_RealValue", "RealValue"),
        ],
    )


def _family_axis():
    from code_index.types import MeasureAxis

    return MeasureAxis(
        axis="measurementQualifier",
        kind="shape",
        role="free",
        prefer=("Nominal", "RealValue"),
    )


def test_collapse_raises_an_unscored_survivor_to_the_familys_scored_member():
    """A collapsed family uses its scored member's value when the survivor lacks a score."""
    from search.engine.retrieval.bound_values import collapse_measure_families

    shapes = (
        _family_shape("t1", score=None),
        _family_shape("t2", score=0.6615),
    )
    collapsed = collapse_measure_families(shapes, _family_index(), (_family_axis(),), named=())
    assert len(collapsed) == 1
    assert collapsed[0].representative_score == pytest.approx(0.6615)


def test_collapse_leaves_an_all_unscored_family_unscored():
    """An all-unscored family keeps an unscored survivor."""
    from search.engine.retrieval.bound_values import collapse_measure_families

    shapes = (
        _family_shape("t1", score=None),
        _family_shape("t2", score=None),
    )
    collapsed = collapse_measure_families(shapes, _family_index(), (_family_axis(),), named=())
    assert len(collapsed) == 1
    assert collapsed[0].representative_score is None


def test_unscored_expansion_shape_still_grades_medium():
    """An unscored retrieved shape grades medium despite available data."""
    from search.engine.region.grounding import _place_confidence

    available, confidence = _place_confidence(
        destination_dcid="country/KEN",
        available_at_place=True,
        representative_score=None,
        retrieved=True,
    )
    assert (available, confidence) == (True, "medium")


def test_not_retrieved_shape_with_data_grades_high():
    """A posted identity with data grades high despite its absent retrieval score."""
    from search.engine.region.grounding import _place_confidence

    available, confidence = _place_confidence(
        destination_dcid="country/KEN",
        available_at_place=True,
        representative_score=None,
        retrieved=False,
    )
    assert (available, confidence) == (True, "high")


def test_not_retrieved_shape_without_data_still_grades_medium():
    """A shape without data grades medium, even when it bypassed retrieval."""
    from search.engine.region.grounding import _grade_confidence

    assert (
        _grade_confidence(score=None, has_data=False, available_at_place=True, retrieved=False)
        == "medium"
    )


def test_a_scored_shape_is_unaffected_by_the_provenance_bit():
    """Measured scores set the band regardless of retrieval provenance."""
    from search.engine.config import settings
    from search.engine.region.grounding import _grade_confidence

    strong = settings().weak_score_threshold + 0.1
    assert _grade_confidence(score=strong, has_data=True, available_at_place=True) == "high"
    assert (
        _grade_confidence(score=strong, has_data=True, available_at_place=True, retrieved=False)
        == "high"
    )


def test_candidate_branch_caps_an_unscored_retrieved_shape_at_medium():
    """An unscored retrieved candidate caps at medium confidence."""
    from search.engine.region.grounding import _place_confidence

    available, confidence = _place_confidence(
        destination_dcid="country/KEN",
        available_at_place=True,
        representative_score=None,
        candidate=True,
        retrieved=True,
    )
    assert (available, confidence) == (True, "medium")


@pytest.mark.parametrize(
    "score,retrieved,expected",
    [
        (0.90, True, "high"),  # scored strong
        (0.10, True, "medium"),  # scored weak
        (None, True, "medium"),  # unscored retrieved (expansion only)
        (None, False, "high"),  # posted identity
    ],
)
def test_candidate_branch_grades_each_provenance_case(score, retrieved, expected):
    """Candidate confidence follows score strength, retrieval status, and availability."""
    from search.engine.region.grounding import _place_confidence

    _available, confidence = _place_confidence(
        destination_dcid="country/KEN",
        available_at_place=True,
        representative_score=score,
        candidate=True,
        retrieved=retrieved,
    )
    assert confidence == expected


@pytest.mark.parametrize(
    ("score", "rival", "sole", "expected"),
    [
        (0.10, True, False, "low"),  # weak beside a structural or nominated rival
        (0.10, False, False, "medium"),  # weak alone
        (0.90, True, False, "high"),  # strong beside a rival
        (0.10, False, True, "high"),  # the only admissible reading with data
    ],
)
def test_a_rival_lowers_and_a_sole_reading_raises_confidence(score, rival, sole, expected):
    """Scores set no status and reach confidence only: a weak score beside a rival is low,
    and the only admissible reading with data is high whatever its score."""
    from search.engine.region.grounding import _place_confidence

    _available, confidence = _place_confidence(
        destination_dcid="country/KEN",
        available_at_place=True,
        representative_score=score,
        candidate=True,
        rival=rival,
        sole_reading=sole,
        retrieved=True,
    )
    assert confidence == expected
