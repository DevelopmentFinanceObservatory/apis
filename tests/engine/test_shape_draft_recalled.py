"""Tests recall status on derived shapes and collapsed measure-family survivors."""

from __future__ import annotations

from code_index import CodeIndex, IndexStamp, StatVarRow
from code_index.types import MeasureAxis
from search.engine.retrieval.bound_values import collapse_measure_families
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft
from tests.engine._discover_fixtures import _std_sv_arcs
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# derive_shapes: recalled vs. topic-expanded-only vs. mixed
# ---------------------------------------------------------------------------

_TOPIC = "dc/topic/Whatever"
_RECALLED_SV = "Recalled_Person_Count"
_TOPIC_MEMBER = "Topic_Member_Person_Count"


def _graph_with_topic(*, member_arcs: dict) -> FakeGraph:
    nodes: dict = {
        _TOPIC: {"arcs": {"relevantVariable": {"nodes": [{"dcid": _TOPIC_MEMBER}]}}},
        _TOPIC_MEMBER: {"label": "topic member", "arcs": member_arcs},
        _RECALLED_SV: {"label": "recalled", "arcs": _std_sv_arcs("Household", "income")},
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def test_recalled_member_shape_reads_true_and_topic_only_shape_reads_false():
    """Only the directly recalled member's shape is marked recalled."""
    graph = _graph_with_topic(member_arcs=_std_sv_arcs("Person", "count"))

    shapes = derive_shapes(
        confirmed_svs=[_TOPIC, _RECALLED_SV],
        graph=graph,
        sv_scores={_TOPIC: 0.90, _RECALLED_SV: 0.50},
        sv_types={_TOPIC: "Topic", _RECALLED_SV: "StatisticalVariable"},
    )

    assert len(shapes) == 2
    recalled_shape = next(s for s in shapes if _RECALLED_SV in (s.sv_arc_facts or {}))
    topic_shape = next(s for s in shapes if _TOPIC_MEMBER in (s.sv_arc_facts or {}))
    assert recalled_shape.recalled is True
    assert topic_shape.recalled is False


def test_mixed_group_with_a_recalled_and_a_topic_expanded_member_reads_true():
    """A mixed group is recalled when one member was directly recalled."""
    # The topic member shares the recalled SV's own five-tuple (Household/
    # income) so both land in the same derive_shapes group.
    graph = _graph_with_topic(member_arcs=_std_sv_arcs("Household", "income"))

    shapes = derive_shapes(
        confirmed_svs=[_TOPIC, _RECALLED_SV],
        graph=graph,
        sv_scores={_TOPIC: 0.90, _RECALLED_SV: 0.50},
        sv_types={_TOPIC: "Topic", _RECALLED_SV: "StatisticalVariable"},
    )

    assert len(shapes) == 1
    shape = shapes[0]
    assert {_RECALLED_SV, _TOPIC_MEMBER} <= set(shape.sv_arc_facts or {})
    assert shape.recalled is True


# ---------------------------------------------------------------------------
# collapse_measure_families: a topic-only survivor inherits a recalled
# sibling's own recall status.
# ---------------------------------------------------------------------------

_ROLES = ("financingProvider", "financingRecipient")

_AXES = (
    MeasureAxis(axis="populationType", kind="shape", role="basis", prefer=()),
    MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=()),
    MeasureAxis(
        axis="measurementQualifier", kind="shape", role="free", prefer=("Nominal", "RealValue")
    ),
)


def _index(statvars: list[StatVarRow]) -> CodeIndex:
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
        statvars=statvars,
    )


def _row(dcid: str, *, meas_qual: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=meas_qual,
        measurement_denominator=None,
        observation_properties=(),
        constraint_properties={},
    )


_MEASURE_ROWS = [_row("t1_nominal", meas_qual="Nominal"), _row("t2_real", meas_qual="RealValue")]


def _shape(shape_id: str, *, meas_qual: str, recalled: bool) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=meas_qual,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=_ROLES,
        recalled=recalled,
    )


def test_collapse_survivor_inherits_a_recalled_siblings_recall_status():
    """The Nominal survivor inherits recall status from its recalled sibling."""
    t1_topic_only = _shape("t1_nominal", meas_qual="Nominal", recalled=False)
    t2_recalled = _shape("t2_real", meas_qual="RealValue", recalled=True)
    index = _index(_MEASURE_ROWS)

    collapsed = collapse_measure_families([t1_topic_only, t2_recalled], index, _AXES)

    assert [s.shape_id for s in collapsed] == ["t1_nominal"]
    assert collapsed[0].recalled is True


def test_collapse_survivor_stays_unrecalled_when_no_sibling_was_recalled():
    """The mirror case: neither family member was recalled, so the survivor
    stays recalled=False."""
    t1_topic_only = _shape("t1_nominal", meas_qual="Nominal", recalled=False)
    t2_topic_only = _shape("t2_real", meas_qual="RealValue", recalled=False)
    index = _index(_MEASURE_ROWS)

    collapsed = collapse_measure_families([t1_topic_only, t2_topic_only], index, _AXES)

    assert [s.shape_id for s in collapsed] == ["t1_nominal"]
    assert collapsed[0].recalled is False
