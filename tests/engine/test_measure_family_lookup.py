"""Lists populated measure partitions and finds exact axis values in the packaged CRS index."""

from __future__ import annotations

from code_index.shared import get_index, get_measure_index
from search.engine.retrieval.bound_values import (
    measure_family_partitions,
    partition_for_values,
)
from search.engine.shape import ShapeDraft
from tests.engine._index_slots import query_for


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
        slot_keys=(),
    )


def test_returns_all_four_populated_partitions_for_the_crs_shape():
    index = get_index()
    axes = get_measure_index().axes

    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))

    assert len(partitions) == 4


def test_returns_an_empty_tuple_for_a_shape_the_index_does_not_carry():
    index = get_index()
    axes = get_measure_index().axes
    shape = ShapeDraft(
        shape_id="unindexed",
        label="unindexed",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="noSuchStatType",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=(),
    )

    assert measure_family_partitions(shape, index, axes, query_for(shape, {})) == ()


def test_partition_for_values_finds_each_populated_partition_by_its_own_tuple():
    index = get_index()
    axes = get_measure_index().axes
    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))

    for partition in partitions:
        found = partition_for_values(partitions, dict(partition.values))
        assert found is partition


def test_partition_for_values_returns_none_for_a_tuple_no_partition_holds():
    index = get_index()
    axes = get_measure_index().axes
    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))
    axis_names = tuple(axis.axis for axis in axes)
    bogus = dict.fromkeys(axis_names, "NoSuchValue")

    assert partition_for_values(partitions, bogus) is None


def test_partition_for_values_distinguishes_none_from_a_value_in_both_directions():
    index = get_index()
    axes = get_measure_index().axes
    partitions = measure_family_partitions(_crs_shape(), index, axes, query_for(_crs_shape(), {}))

    # Flow partitions omit this axis; commitment partitions declare a value.
    none_partition = next(
        p for p in partitions if dict(p.values)["financialObligationType"] is None
    )
    none_values = dict(none_partition.values)
    assert partition_for_values(partitions, none_values) is none_partition

    valued_partition = next(
        p for p in partitions if dict(p.values)["financialObligationType"] is not None
    )
    valued_values = dict(valued_partition.values)
    assert none_values["financialObligationType"] != valued_values["financialObligationType"]

    # Adding the commitment value to a flow tuple yields no partition.
    swapped = {**none_values, "financialObligationType": valued_values["financialObligationType"]}
    assert partition_for_values(partitions, swapped) is None

    # Clearing the commitment value from its obligation tuple also matches nothing.
    forced_none = {**valued_values, "financialObligationType": None}
    assert partition_for_values(partitions, forced_none) is None
