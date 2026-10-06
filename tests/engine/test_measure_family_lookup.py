"""Lists populated measure partitions and finds exact axis values in the packaged CRS index."""

from __future__ import annotations

import pytest

from code_index.shared import get_index, get_measure_index
from search.engine.retrieval.bound_values import (
    measure_family_partitions,
    partition_for_values,
    shape_query,
)
from search.engine.shape import ShapeDraft
from tests.engine._index_slots import crs_flow_slot_keys, index_slot_keys, query_for

_CRS = "provenance/OECDCreditorReportingSystem"


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


@pytest.mark.parametrize(
    ("population_type", "measured_property"),
    [
        ("FinancialFlow", "amount"),
        ("FinancialFlow", "grantEquivalent"),
        ("FinancialObligation", "amount"),
    ],
)
@pytest.mark.parametrize(
    "role_keys", [(), ("origin",), ("destination",), ("origin", "destination")]
)
def test_a_crs_shape_reads_every_crs_row_of_its_measure_siblings(
    population_type, measured_property, role_keys
):
    """Each sibling five-tuple elects its rows apart, and every CRS shape still reads the
    CRS rows of all four siblings whole, whatever places the query binds, and no DAC1 row
    the same five-tuples carry: CRS flows, net and gross, grant equivalents and
    commitments."""
    index = get_index()
    shape = ShapeDraft(
        shape_id="crs",
        label="crs",
        pop_type_dcid=population_type,
        meas_prop_dcid=measured_property,
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=index_slot_keys(
            population_type=population_type,
            measured_property=measured_property,
            measurement_qualifier="Nominal",
        ),
        declared_roles=("financingProvider", "financingRecipient"),
    )
    query = shape_query(shape, [], role_keys)

    partitions = measure_family_partitions(shape, index, get_measure_index().axes, query)

    assert len(partitions) == 4
    assert {dcid for partition in partitions for dcid in partition.members} == {
        row.dcid
        for row in index.statvars
        if _CRS in row.provenances
        and row.measured_property in ("amount", "grantEquivalent")
        and row.measurement_denominator is None
    }
