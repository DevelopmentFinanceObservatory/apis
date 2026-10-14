"""Tests declared measure election and family collapse.

Synthetic axes prove `role="basis"` follows declarations, not field names."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import pytest

from code_index import CodeIndex, IndexStamp, StatVarRow
from code_index.types import MeasureAxis, MeasurePartitionLabel
from search.engine.qualifier_match import QualifierMatch
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    PartitionLabelMatch,
    _measure_axis_value,
    collapse_measure_families,
    elect_default_partition,
    label_for_partition,
    shape_carries_measure_value,
    sibling_partitions,
)
from search.engine.shape import ShapeDraft

# Eight partitions across six axes include the grant-equivalent carve-out.

_MEASURE_AXES = (
    MeasureAxis(
        axis="populationType",
        kind="shape",
        role="free",
        prefer=("FinancialFlow", "FinancialObligation"),
    ),
    MeasureAxis(
        axis="measuredProperty", kind="shape", role="basis", prefer=("amount", "grantEquivalent")
    ),
    MeasureAxis(
        axis="measurementQualifier", kind="shape", role="free", prefer=("Nominal", "RealValue")
    ),
    MeasureAxis(axis="financialFlowType", kind="constraint", role="free", prefer=("Disbursement",)),
    MeasureAxis(
        axis="financialObligationType", kind="constraint", role="free", prefer=("Commitment",)
    ),
    MeasureAxis(axis="nettingBasis", kind="constraint", role="free", prefer=("Gross", "Net")),
)


def _sv(
    dcid: str,
    *,
    population_type: str,
    measured_property: str,
    measurement_qualifier: str,
    **constraints: str,
) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type=population_type,
        measured_property=measured_property,
        stat_type="measuredValue",
        measurement_qualifier=measurement_qualifier,
        measurement_denominator=None,
        observation_properties=(),
        constraint_properties=constraints,
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


_WORKED_EXAMPLE = [
    _sv(
        "SV_FF_amount_Nominal_Disb_Gross",
        population_type="FinancialFlow",
        measured_property="amount",
        measurement_qualifier="Nominal",
        financialFlowType="Disbursement",
        nettingBasis="Gross",
    ),
    _sv(
        "SV_FF_amount_RealValue_Disb_Gross",
        population_type="FinancialFlow",
        measured_property="amount",
        measurement_qualifier="RealValue",
        financialFlowType="Disbursement",
        nettingBasis="Gross",
    ),
    _sv(
        "SV_FF_amount_Nominal_Disb_Net",
        population_type="FinancialFlow",
        measured_property="amount",
        measurement_qualifier="Nominal",
        financialFlowType="Disbursement",
        nettingBasis="Net",
    ),
    _sv(
        "SV_FF_amount_RealValue_Disb_Net",
        population_type="FinancialFlow",
        measured_property="amount",
        measurement_qualifier="RealValue",
        financialFlowType="Disbursement",
        nettingBasis="Net",
    ),
    _sv(
        "SV_FO_amount_Nominal_Commit",
        population_type="FinancialObligation",
        measured_property="amount",
        measurement_qualifier="Nominal",
        financialObligationType="Commitment",
    ),
    _sv(
        "SV_FO_amount_RealValue_Commit",
        population_type="FinancialObligation",
        measured_property="amount",
        measurement_qualifier="RealValue",
        financialObligationType="Commitment",
    ),
    _sv(
        "SV_FF_grantEq_Nominal_Disb",
        population_type="FinancialFlow",
        measured_property="grantEquivalent",
        measurement_qualifier="Nominal",
        financialFlowType="Disbursement",
    ),
    _sv(
        "SV_FF_grantEq_RealValue_Disb",
        population_type="FinancialFlow",
        measured_property="grantEquivalent",
        measurement_qualifier="RealValue",
        financialFlowType="Disbursement",
    ),
]


# Literal partitions isolate the election; `collapse_measure_families` builds the same values
# from ShapeDraft fields.


def _partition(values: dict[str, str | None], *members: str) -> MeasurePartition:
    return MeasurePartition(values=tuple(values.items()), members=members)


_WORKED_EXAMPLE_PARTITIONS = (
    _partition(
        {
            "populationType": "FinancialFlow",
            "measuredProperty": "amount",
            "measurementQualifier": "Nominal",
            "financialFlowType": "Disbursement",
            "financialObligationType": None,
            "nettingBasis": "Gross",
        },
        "SV_FF_amount_Nominal_Disb_Gross",
    ),
    _partition(
        {
            "populationType": "FinancialFlow",
            "measuredProperty": "amount",
            "measurementQualifier": "RealValue",
            "financialFlowType": "Disbursement",
            "financialObligationType": None,
            "nettingBasis": "Gross",
        },
        "SV_FF_amount_RealValue_Disb_Gross",
    ),
    _partition(
        {
            "populationType": "FinancialFlow",
            "measuredProperty": "amount",
            "measurementQualifier": "Nominal",
            "financialFlowType": "Disbursement",
            "financialObligationType": None,
            "nettingBasis": "Net",
        },
        "SV_FF_amount_Nominal_Disb_Net",
    ),
    _partition(
        {
            "populationType": "FinancialFlow",
            "measuredProperty": "amount",
            "measurementQualifier": "RealValue",
            "financialFlowType": "Disbursement",
            "financialObligationType": None,
            "nettingBasis": "Net",
        },
        "SV_FF_amount_RealValue_Disb_Net",
    ),
    _partition(
        {
            "populationType": "FinancialObligation",
            "measuredProperty": "amount",
            "measurementQualifier": "Nominal",
            "financialFlowType": None,
            "financialObligationType": "Commitment",
            "nettingBasis": None,
        },
        "SV_FO_amount_Nominal_Commit",
    ),
    _partition(
        {
            "populationType": "FinancialObligation",
            "measuredProperty": "amount",
            "measurementQualifier": "RealValue",
            "financialFlowType": None,
            "financialObligationType": "Commitment",
            "nettingBasis": None,
        },
        "SV_FO_amount_RealValue_Commit",
    ),
    _partition(
        {
            "populationType": "FinancialFlow",
            "measuredProperty": "grantEquivalent",
            "measurementQualifier": "Nominal",
            "financialFlowType": "Disbursement",
            "financialObligationType": None,
            "nettingBasis": None,
        },
        "SV_FF_grantEq_Nominal_Disb",
    ),
    _partition(
        {
            "populationType": "FinancialFlow",
            "measuredProperty": "grantEquivalent",
            "measurementQualifier": "RealValue",
            "financialFlowType": "Disbursement",
            "financialObligationType": None,
            "nettingBasis": None,
        },
        "SV_FF_grantEq_RealValue_Disb",
    ),
)


def test_worked_example_elects_gross_nominal_disbursement_with_five_amount_siblings():
    election = elect_default_partition(_WORKED_EXAMPLE_PARTITIONS, _MEASURE_AXES)

    assert election is not None
    assert election.elected.members == ("SV_FF_amount_Nominal_Disb_Gross",)
    sibling_members = {dcid for sibling in election.siblings for dcid in sibling.members}
    assert sibling_members == {
        "SV_FF_amount_RealValue_Disb_Gross",
        "SV_FF_amount_Nominal_Disb_Net",
        "SV_FF_amount_RealValue_Disb_Net",
        "SV_FO_amount_Nominal_Commit",
        "SV_FO_amount_RealValue_Commit",
    }
    # Neither grantEquivalent partition -- measuredProperty is the one
    # role="basis" axis, and the election's own value for it is "amount".
    assert "SV_FF_grantEq_Nominal_Disb" not in sibling_members
    assert "SV_FF_grantEq_RealValue_Disb" not in sibling_members


def test_no_partitions_returns_no_election():
    # A deployment with no declared measure family reaches this with an
    # empty partitions sequence -- not an empty election, no election at all.
    assert elect_default_partition((), _MEASURE_AXES) is None


def test_most_significant_axis_decides_over_a_later_one():
    # SV_A wins the first (most significant) axis but loses the second;
    # SV_B is the reverse. The first axis decides regardless of the second.
    axes = (
        MeasureAxis(
            axis="populationType",
            kind="shape",
            role="free",
            prefer=("FinancialFlow", "FinancialObligation"),
        ),
        MeasureAxis(
            axis="measurementQualifier", kind="shape", role="free", prefer=("Nominal", "RealValue")
        ),
    )
    sv_a = _partition(
        {"populationType": "FinancialFlow", "measurementQualifier": "RealValue"}, "SV_A"
    )
    sv_b = _partition(
        {"populationType": "FinancialObligation", "measurementQualifier": "Nominal"}, "SV_B"
    )

    election = elect_default_partition((sv_a, sv_b), axes)

    assert election is not None
    assert election.elected.members == ("SV_A",)


def test_value_absent_from_the_prefer_list_sorts_after_every_preferred_value():
    # A partition's own value on an axis, when outside that axis's own
    # prefer list -- None included, a member that never declares the axis at
    # all -- never outranks one that IS on the list, on that axis alone.
    axis = MeasureAxis(axis="nettingBasis", kind="constraint", role="free", prefer=("Gross", "Net"))
    preferred = _partition({"nettingBasis": "Gross"}, "SV_PREFERRED")
    unlisted_value = _partition({"nettingBasis": "Other"}, "SV_UNLISTED_VALUE")
    absent = _partition({"nettingBasis": None}, "SV_ABSENT")

    election = elect_default_partition((unlisted_value, absent, preferred), (axis,))

    assert election is not None
    assert election.elected.members == ("SV_PREFERRED",)


# Swap the CRS roles so measurementQualifier is basis and measuredProperty is free.
# This proves sibling rules follow declarations, not axis names.

_S3_AXES = tuple(
    axis.__class__(axis=axis.axis, kind=axis.kind, prefer=axis.prefer, role=role)
    for axis, role in zip(
        _MEASURE_AXES,
        ("free", "free", "basis", "free", "free", "free"),
        strict=True,
    )
)


def test_s3_inverted_basis_grant_equivalent_becomes_a_sibling_of_amount():
    election = elect_default_partition(_WORKED_EXAMPLE_PARTITIONS, _S3_AXES)

    assert election is not None
    assert election.elected.members == ("SV_FF_amount_Nominal_Disb_Gross",)
    sibling_members = {dcid for sibling in election.siblings for dcid in sibling.members}
    assert sibling_members == {
        "SV_FF_amount_Nominal_Disb_Net",
        "SV_FO_amount_Nominal_Commit",
        "SV_FF_grantEq_Nominal_Disb",
    }
    # `measurementQualifier` is basis, so only Nominal partitions qualify.
    assert not any("RealValue" in dcid for dcid in sibling_members)


# ShapeDrafts group by five-tuples. Constraint axes such as `nettingBasis` affect election
# without splitting shapes.


def _shape(
    shape_id: str,
    *,
    pop_type_dcid: str,
    meas_prop_dcid: str,
    meas_qual_dcid: str | None,
    stat_type_dcid: str = "measuredValue",
    meas_denom_dcid: str | None = None,
    meas_denom_unresolved: bool = False,
    declared_roles: tuple[str, ...] | None = None,
    members: Sequence[str] = (),
    representative_score: float = 1.0,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=pop_type_dcid,
        meas_prop_dcid=meas_prop_dcid,
        stat_type_dcid=stat_type_dcid,
        meas_qual_dcid=meas_qual_dcid,
        meas_denom_dcid=meas_denom_dcid,
        meas_denom_unresolved=meas_denom_unresolved,
        declared_roles=declared_roles,
        slot_keys=(),
        sv_arc_facts=({dcid: {} for dcid in members} if members else None),
        representative_score=representative_score,
    )


def _worked_example_shapes() -> dict[str, ShapeDraft]:
    """Build six shapes with two amount qualifiers per population type and two
    grant-equivalent qualifiers."""
    return {
        "t1": _shape(
            "t1_ff_amount_nominal",
            pop_type_dcid="FinancialFlow",
            meas_prop_dcid="amount",
            meas_qual_dcid="Nominal",
        ),
        "t2": _shape(
            "t2_ff_amount_realvalue",
            pop_type_dcid="FinancialFlow",
            meas_prop_dcid="amount",
            meas_qual_dcid="RealValue",
        ),
        "t3": _shape(
            "t3_fo_amount_nominal",
            pop_type_dcid="FinancialObligation",
            meas_prop_dcid="amount",
            meas_qual_dcid="Nominal",
        ),
        "t4": _shape(
            "t4_fo_amount_realvalue",
            pop_type_dcid="FinancialObligation",
            meas_prop_dcid="amount",
            meas_qual_dcid="RealValue",
        ),
        "t5": _shape(
            "t5_ff_granteq_nominal",
            pop_type_dcid="FinancialFlow",
            meas_prop_dcid="grantEquivalent",
            meas_qual_dcid="Nominal",
        ),
        "t6": _shape(
            "t6_ff_granteq_realvalue",
            pop_type_dcid="FinancialFlow",
            meas_prop_dcid="grantEquivalent",
            meas_qual_dcid="RealValue",
        ),
    }


def test_six_shapes_differing_only_by_measure_collapse_to_the_elected_one():
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)

    collapsed = collapse_measure_families(
        [shapes["t1"], shapes["t2"], shapes["t3"], shapes["t4"], shapes["t5"], shapes["t6"]],
        index,
        _MEASURE_AXES,
    )

    assert collapsed == (shapes["t1"], shapes["t5"])


def test_grant_equivalent_family_collapses_to_its_own_elected_member():
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)

    collapsed = collapse_measure_families([shapes["t5"], shapes["t6"]], index, _MEASURE_AXES)

    # The basis axis separates grant-equivalent from amount; both qualifiers still collapse.
    assert collapsed == (shapes["t5"],)


def test_a_family_collapses_whatever_arc_order_its_shapes_declare_their_roles_in():
    shapes = _worked_example_shapes()
    roles = ("financingProvider", "financingRecipient")
    nominal = replace(shapes["t1"], declared_roles=roles)
    real = replace(shapes["t2"], declared_roles=roles[::-1])

    collapsed = collapse_measure_families([nominal, real], _index(_WORKED_EXAMPLE), _MEASURE_AXES)

    assert collapsed == (nominal,)


def test_collapse_ignores_which_members_the_sv_arc_facts_sample_happens_to_hold():
    # Gross, Net, and empty recall samples give the same result;
    # membership follows the five-tuple.
    index = _index(_WORKED_EXAMPLE)
    common = {
        "pop_type_dcid": "FinancialFlow",
        "meas_prop_dcid": "amount",
        "meas_qual_dcid": "Nominal",
    }
    t1_gross = _shape("t1", members=["SV_FF_amount_Nominal_Disb_Gross"], **common)
    t1_net = _shape("t1", members=["SV_FF_amount_Nominal_Disb_Net"], **common)
    t1_empty = _shape("t1", members=[], **common)

    assert collapse_measure_families([t1_gross], index, _MEASURE_AXES) == (t1_gross,)
    assert collapse_measure_families([t1_net], index, _MEASURE_AXES) == (t1_net,)
    assert collapse_measure_families([t1_empty], index, _MEASURE_AXES) == (t1_empty,)


def test_shape_absent_from_index_passes_through_even_when_it_would_rank_last():
    # `Peculiar` is absent from the index, so pass-through follows missing membership, not rank.
    unindexed = _shape(
        "unindexed",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        meas_qual_dcid="Peculiar",
    )
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)

    collapsed = collapse_measure_families([unindexed, shapes["t2"]], index, _MEASURE_AXES)

    assert collapsed == (unindexed, shapes["t2"])


def test_family_differing_on_a_non_measure_field_does_not_collapse_together():
    # Different `stat_type_dcid` values keep a rate and a total in separate families.
    rate_row = StatVarRow(
        dcid="SV_MEASURED",
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=(),
        constraint_properties={},
    )
    median_row = StatVarRow(
        dcid="SV_MEDIAN",
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="medianValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=(),
        constraint_properties={},
    )
    measured = _shape(
        "measured",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        meas_qual_dcid="Nominal",
        stat_type_dcid="measuredValue",
    )
    median = _shape(
        "median",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        meas_qual_dcid="Nominal",
        stat_type_dcid="medianValue",
    )
    index = _index([rate_row, median_row])

    collapsed = collapse_measure_families([measured, median], index, _MEASURE_AXES)

    assert collapsed == (measured, median)


def test_no_declared_axes_leaves_shapes_unchanged():
    # An empty axis declaration preserves shapes and order.
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)

    collapsed = collapse_measure_families([shapes["t1"], shapes["t2"]], index, ())

    assert collapsed == (shapes["t1"], shapes["t2"])


def test_synthetic_axes_collapse_using_their_own_preference_order_not_crs():
    # This synthetic order prefers FinancialObligation, opposite to CRS's order for t1 and t3.
    axes = (
        MeasureAxis(
            axis="populationType",
            kind="shape",
            role="free",
            prefer=("FinancialObligation", "FinancialFlow"),
        ),
    )
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)

    collapsed = collapse_measure_families([shapes["t1"], shapes["t3"]], index, axes)

    assert collapsed == (shapes["t3"],)


def test_non_empty_input_never_collapses_to_empty_output():
    axes = (
        MeasureAxis(
            axis="populationType",
            kind="shape",
            role="free",
            prefer=("FinancialFlow", "FinancialObligation"),
        ),
    )
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)

    collapsed = collapse_measure_families(
        [shapes["t1"], shapes["t2"], shapes["t3"], shapes["t4"]], index, axes
    )

    assert len(collapsed) == 1


def test_shapes_spanning_a_basis_constraint_axis_are_never_collapsed():
    # `nettingBasis` lies outside the five-tuple, so a basis declaration blocks shape collapse.
    axes = (
        *_MEASURE_AXES,
        MeasureAxis(axis="nettingBasis", kind="constraint", role="basis", prefer=("Gross", "Net")),
    )
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)

    collapsed = collapse_measure_families([shapes["t1"], shapes["t2"]], index, axes)

    assert collapsed == (shapes["t1"], shapes["t2"])


def test_collapse_preserves_order_so_a_caller_indexing_the_result_finds_the_right_shape():
    # Bind prompt indices must continue to identify the same shapes after collapse.
    before = _shape(
        "before",
        pop_type_dcid="FinancialObligation",
        meas_prop_dcid="grantEquivalent",
        meas_qual_dcid="Nominal",
    )
    after = _shape(
        "after",
        pop_type_dcid="FinancialObligation",
        meas_prop_dcid="grantEquivalent",
        meas_qual_dcid="RealValue",
    )
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)

    collapsed = collapse_measure_families(
        [before, shapes["t2"], shapes["t1"], after], index, _MEASURE_AXES
    )

    assert collapsed == (before, shapes["t1"], after)
    assert collapsed[1].shape_id == "t1_ff_amount_nominal"


# Survivors inherit their family's best score, keeping them within the bind cap.
# `_unrelated` shapes use five-tuples outside the fixture index.


def _unrelated(i: int, *, representative_score: float) -> ShapeDraft:
    return _shape(
        f"unrelated{i}",
        pop_type_dcid="Person",
        meas_prop_dcid=f"unrelatedProp{i}",
        meas_qual_dcid=None,
        representative_score=representative_score,
    )


def test_survivor_scores_as_family_max_and_reaches_the_cap_past_crowding_unrelated_shapes():
    # t2 scores 0.6615 and wins retrieval, while preferred t1 scores 0.6605.
    # Eight unrelated shapes fit between them and fill K=8.
    # t1 must inherit the top score to stay in the cap.
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    t1 = replace(shapes["t1"], representative_score=0.6605)
    t2 = replace(shapes["t2"], representative_score=0.6615)
    unrelated = [_unrelated(i, representative_score=0.6614 - i * 0.0001) for i in range(8)]
    ranked_shapes = [
        t2,
        *unrelated,
        t1,
    ]  # already score-descending, as _phase_prune_and_rank hands in

    collapsed = collapse_measure_families(ranked_shapes, index, _MEASURE_AXES)

    # The elected identity stays Nominal; only its score changes.
    assert collapsed[0].shape_id == "t1_ff_amount_nominal"
    assert collapsed[0].representative_score == pytest.approx(0.6615)
    assert len(collapsed) == 9  # ten in, one family member folded away


def test_preferred_member_already_top_scored_is_unchanged():
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    t1 = replace(shapes["t1"], representative_score=0.9)
    t2 = replace(shapes["t2"], representative_score=0.5)

    collapsed = collapse_measure_families([t1, t2], index, _MEASURE_AXES)

    assert collapsed[0].shape_id == "t1_ff_amount_nominal"
    assert collapsed[0].representative_score == pytest.approx(0.9)
    assert len(collapsed) == 1


def test_collapsed_list_stays_score_descending():
    # Near-neighbour lookup stops at the first distinct concept, so the list must stay score-sorted.
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    t1 = replace(shapes["t1"], representative_score=0.6605)
    t2 = replace(shapes["t2"], representative_score=0.6615)
    unrelated = [_unrelated(i, representative_score=0.6614 - i * 0.0001) for i in range(8)]
    ranked_shapes = [t2, *unrelated, t1]

    collapsed = collapse_measure_families(ranked_shapes, index, _MEASURE_AXES)

    scores = [s.representative_score for s in collapsed]
    assert scores == sorted(scores, reverse=True)


# Query matches narrow each group before election; an unmatched group keeps its default partition.


def _measure_match(
    code_system: str, code_dcid: str, matched_text: str | None = None
) -> QualifierMatch:
    return QualifierMatch(
        code_system=code_system,
        code_dcid=code_dcid,
        label=code_dcid,
        source="label",
        matched_text=code_dcid if matched_text is None else matched_text,
    )


def test_named_constraint_value_steers_the_collapse_across_a_shape_axis():
    # The `commitments` filter selects FinancialObligation through `financialObligationType`,
    # which is absent from `_family_key` (#743).
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    named = (_measure_match("financialObligationType", "Commitment", "commitments"),)

    collapsed = collapse_measure_families(
        [shapes["t1"], shapes["t2"], shapes["t3"], shapes["t4"]],
        index,
        _MEASURE_AXES,
        named=named,
    )

    assert collapsed == (shapes["t3"],)


def test_named_shape_axis_value_steers_the_collapse_to_the_matching_qualifier():
    # The qualifier filter elects RealValue instead of the Nominal default.
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    named = (_measure_match("measurementQualifier", "RealValue", "in real terms"),)

    collapsed = collapse_measure_families(
        [shapes["t1"], shapes["t2"], shapes["t3"], shapes["t4"]],
        index,
        _MEASURE_AXES,
        named=named,
    )

    assert collapsed == (shapes["t2"],)


def test_query_naming_no_measure_collapses_exactly_as_the_undecorated_call_does():
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    group = [shapes["t1"], shapes["t2"], shapes["t3"], shapes["t4"]]

    baseline = collapse_measure_families(group, index, _MEASURE_AXES)
    explicit_empty = collapse_measure_families(group, index, _MEASURE_AXES, named=())

    assert baseline == explicit_empty == (shapes["t1"],)


def test_basis_axis_match_elects_exactly_as_no_match_does():
    # A `measuredProperty` basis match does not steer collapse.
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    group = [shapes["t1"], shapes["t2"], shapes["t3"], shapes["t4"]]
    named = (_measure_match("measuredProperty", "grantEquivalent"),)

    with_basis_match = collapse_measure_families(group, index, _MEASURE_AXES, named=named)
    with_nothing_named = collapse_measure_families(group, index, _MEASURE_AXES)

    assert with_basis_match == with_nothing_named == (shapes["t1"],)


def test_named_value_no_family_member_carries_elects_the_default_and_never_empties():
    # Grant-equivalent rows omit nettingBasis, so unmatched `Net` leaves Nominal as the default.
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    named = (_measure_match("nettingBasis", "Net", "net"),)

    collapsed = collapse_measure_families(
        [shapes["t5"], shapes["t6"]], index, _MEASURE_AXES, named=named
    )

    assert collapsed == (shapes["t5"],)


def test_two_values_of_one_axis_in_one_variable_are_dropped_and_elect_the_default():
    # Conflicting Gross and Net matches leave the axis unbound for election.
    shapes = _worked_example_shapes()
    index = _index(_WORKED_EXAMPLE)
    group = [shapes["t1"], shapes["t2"], shapes["t3"], shapes["t4"]]
    named = (
        _measure_match("nettingBasis", "Gross", "gross"),
        _measure_match("nettingBasis", "Net", "net"),
    )

    collapsed = collapse_measure_families(group, index, _MEASURE_AXES, named=named)

    assert collapsed == (shapes["t1"],)


def test_shape_carries_measure_value_reads_the_index_not_just_the_shape():
    index = _index(_WORKED_EXAMPLE)
    shapes = _worked_example_shapes()
    netting_axis = next(axis for axis in _MEASURE_AXES if axis.axis == "nettingBasis")

    # `nettingBasis` is outside t1's five-tuple, so the index supplies its values.
    assert shape_carries_measure_value(shapes["t1"], index, netting_axis, "Net")
    assert shape_carries_measure_value(shapes["t1"], index, netting_axis, "Gross")

    # No grant-equivalent row carries `nettingBasis`.
    assert not shape_carries_measure_value(shapes["t5"], index, netting_axis, "Net")


# Authored labels match subsets of a partition's populated axes; synthetic values pin subset,
# largest-match, and tie behavior.


def test_subset_label_matches_and_a_label_naming_an_absent_axis_does_not():
    partition = _partition({"color": "Red", "size": "Large"}, "SV_A")
    subset_label = MeasurePartitionLabel(axis_values=(("color", "Red"),), label="red")
    label_for_absent_axis = MeasurePartitionLabel(axis_values=(("finish", "Matte"),), label="matte")

    match = label_for_partition(partition, (subset_label, label_for_absent_axis))

    assert match == PartitionLabelMatch(label="red", uncovered=(("size", "Large"),))


def test_largest_match_wins_over_a_smaller_one_that_also_matches():
    partition = _partition({"color": "Red", "size": "Large"}, "SV_A")
    smaller = MeasurePartitionLabel(axis_values=(("color", "Red"),), label="red")
    larger = MeasurePartitionLabel(
        axis_values=(("color", "Red"), ("size", "Large")), label="large red"
    )

    match = label_for_partition(partition, (smaller, larger))

    assert match == PartitionLabelMatch(label="large red", uncovered=())


def test_two_equal_size_matches_raise():
    partition = _partition({"color": "Red", "size": "Large"}, "SV_A")
    first = MeasurePartitionLabel(axis_values=(("color", "Red"),), label="red")
    second = MeasurePartitionLabel(axis_values=(("size", "Large"),), label="large")

    with pytest.raises(ValueError):
        label_for_partition(partition, (first, second))


def test_uncovered_pairs_come_back_in_declaration_order():
    partition = _partition({"color": "Red", "size": "Large", "finish": "Matte"}, "SV_A")
    label = MeasurePartitionLabel(axis_values=(("size", "Large"),), label="large")

    match = label_for_partition(partition, (label,))

    assert match is not None
    assert match.uncovered == (("color", "Red"), ("finish", "Matte"))


def test_no_matching_label_returns_the_absence_not_an_exception():
    partition = _partition({"color": "Red"}, "SV_A")
    unrelated = MeasurePartitionLabel(axis_values=(("finish", "Matte"),), label="matte")

    assert label_for_partition(partition, (unrelated,)) is None


def test_none_valued_axes_are_excluded_before_matching():
    # Labels match populated axes; this partition has no `size` value.
    partition = _partition({"color": "Red", "size": None}, "SV_A")
    label_naming_the_absent_axis = MeasurePartitionLabel(
        axis_values=(("color", "Red"), ("size", "Large")), label="large red"
    )
    label_on_the_present_axis_only = MeasurePartitionLabel(
        axis_values=(("color", "Red"),), label="red"
    )

    match = label_for_partition(
        partition, (label_naming_the_absent_axis, label_on_the_present_axis_only)
    )

    assert match == PartitionLabelMatch(label="red", uncovered=())


# Four packaged partitions each have one label; `measurementQualifier` remains uncovered.


def _real_partitions() -> tuple[MeasurePartition, ...]:
    from code_index.shared import get_index, get_measure_index

    index = get_index()
    axes = get_measure_index().axes
    axis_names = tuple(axis.axis for axis in axes)
    grouped: dict[tuple[str | None, ...], list[str]] = defaultdict(list)
    for row in index.statvars:
        grouped[tuple(_measure_axis_value(row, axis) for axis in axes)].append(row.dcid)
    return tuple(
        MeasurePartition(values=tuple(zip(axis_names, key, strict=True)), members=tuple(dcids))
        for key, dcids in grouped.items()
    )


def test_real_index_every_populated_partition_resolves_to_exactly_one_label():
    """Each partition has one label, and no two partitions of one basis family share it.

    Labels state no price basis, so a family holding both a Nominal and a
    RealValue partition would show two identical entries in the measure switch.
    """
    from code_index.labels import compose_measure_label
    from code_index.shared import get_measure_index

    measure_index = get_measure_index()
    partitions = _real_partitions()

    assert len(partitions) == 9
    composed: dict[MeasurePartition, str] = {}
    for partition in partitions:
        match = label_for_partition(partition, measure_index.partition_labels)
        assert match is not None
        assert [axis for axis, _ in match.uncovered] == ["measurementQualifier"]
        composed[partition] = compose_measure_label(match, measure_index.values)
    for partition in partitions:
        siblings = sibling_partitions(partition, partitions, measure_index.axes)
        assert composed[partition] not in {composed[sibling] for sibling in siblings}
