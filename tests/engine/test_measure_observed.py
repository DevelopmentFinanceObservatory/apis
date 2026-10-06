"""Tests observed measure partitions for resolved members against the packaged CRS index.

The FinancialFlow/amount/measuredValue/Nominal family has four populated partitions.
"""

from __future__ import annotations

from code_index.shared import get_index, get_measure_index
from search.engine.retrieval.bound_values import (
    measure_family_partitions,
    observed_measure_family,
    observed_measure_values,
    sibling_partitions,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts, IndexMemberFacts
from search.engine.shape import ShapeDraft
from tests.engine._index_slots import crs_flow_slot_keys, query_for


def _crs_shape() -> ShapeDraft:
    index = get_index()
    flow_row = next(row for row in index.statvars if row.population_type == "FinancialFlow")
    return ShapeDraft(
        shape_id="crs-amount",
        label="crs-amount",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid=flow_row.stat_type,
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=crs_flow_slot_keys(),
    )


def _members_for(dcids: tuple[str, ...]) -> list[IndexMemberFacts]:
    index = get_index()
    rows_by_dcid = {row.dcid: row for row in index.statvars}
    return [IndexMemberFacts(row=rows_by_dcid[dcid]) for dcid in dcids]


def _arcs_from_real_row(dcid: str) -> dict:
    """Build node arcs from a packaged row to compare arcs and index member facts."""
    index = get_index()
    row = next(r for r in index.statvars if r.dcid == dcid)
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }
    if row.measurement_qualifier:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": row.measurement_qualifier}]}
    if row.measurement_denominator:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": row.measurement_denominator}]}
    return arcs


def test_observed_values_match_each_of_the_four_populated_partitions_taken_as_served():
    index = get_index()
    axes = get_measure_index().axes
    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))
    assert len(partitions) == 4

    for partition in partitions:
        members = _members_for(partition.members)
        observed = observed_measure_values(axes, members)
        assert observed is not None
        assert dict(observed.values) == dict(partition.values)


def test_observed_measure_family_finds_the_family_partition_object_and_its_siblings():
    index = get_index()
    axes = get_measure_index().axes
    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))

    for partition in partitions:
        members = _members_for(partition.members)
        observed, siblings = observed_measure_family(
            _crs_shape(), index, axes, query_for(_crs_shape(), {}), members
        )
        assert observed is not None
        assert dict(observed.values) == dict(partition.values)
        expected_siblings = sibling_partitions(partition, partitions, axes)
        assert {s.values for s in siblings} == {s.values for s in expected_siblings}
        # observed's own siblings never include a partition equal to itself.
        assert observed.values not in {s.values for s in siblings}


def test_disagreeing_members_publish_none_for_both():
    index = get_index()
    axes = get_measure_index().axes
    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))
    # Members drawn from two distinct partitions: they cannot share every
    # axis value, so there is no single partition to observe.
    a, b = partitions[0], partitions[1]
    members = _members_for((a.members[0], b.members[0]))

    assert observed_measure_values(axes, members) is None
    assert observed_measure_family(
        _crs_shape(), index, axes, query_for(_crs_shape(), {}), members
    ) == (None, ())


def test_empty_members_publish_none():
    axes = get_measure_index().axes
    assert observed_measure_values(axes, []) is None
    assert observed_measure_family(
        _crs_shape(), get_index(), axes, query_for(_crs_shape(), {}), []
    ) == (None, ())


def test_a_shape_absent_from_the_index_publishes_none_regardless_of_members():
    """An unindexed shape has no measure family, even when its members share axis values."""
    index = get_index()
    axes = get_measure_index().axes
    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))
    members = _members_for(partitions[0].members)
    unindexed_shape = ShapeDraft(
        shape_id="unindexed",
        label="unindexed",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="noSuchStatType",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(),
    )

    assert observed_measure_family(
        unindexed_shape, index, axes, query_for(unindexed_shape, {}), members
    ) == (None, ())


def test_arcs_member_facts_and_index_member_facts_observe_the_same_partition():
    """Arcs and index facts for one StatVar must observe identical axis values."""
    index = get_index()
    axes = get_measure_index().axes
    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))
    partition = partitions[0]

    index_members = _members_for(partition.members)
    arcs_members = [ArcsMemberFacts(arcs=_arcs_from_real_row(dcid)) for dcid in partition.members]

    from_index = observed_measure_values(axes, index_members)
    from_arcs = observed_measure_values(axes, arcs_members)
    assert from_index is not None
    assert from_arcs is not None
    assert dict(from_index.values) == dict(from_arcs.values)
