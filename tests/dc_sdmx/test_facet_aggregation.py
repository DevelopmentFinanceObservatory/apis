"""Test facet aggregation and its consistency checks."""

from __future__ import annotations

import pytest

from dc_sdmx.errors import SdmxResponseError
from dc_sdmx.results import SdmxFacetAggregate, SdmxObservation, aggregate_observations_by_facet


def _row(
    *,
    variable: str = "MalariaGrantDisbursements",
    role_values: dict[str, str] | None = None,
    unit: str | None = "USD",
    measurement_method: str | None = "GrantEquivalent",
    observation_period: str | None = "P1Y",
    provenance: str | None = "OECD_DAC2A",
    time_period: str | None = "2020",
    value: str = "100",
    scaling_factor: str | None = "0",
    facet_id: str | None = "facet1",
) -> SdmxObservation:
    return SdmxObservation(
        variable=variable,
        role_values=role_values or {"financingRecipient": "country/KEN"},
        unit=unit,
        measurement_method=measurement_method,
        observation_period=observation_period,
        provenance=provenance,
        time_period=time_period,
        value=value,
        scaling_factor=scaling_factor,
        facet_id=facet_id,
    )


def test_rows_sharing_a_facet_id_aggregate_into_one_entry():
    rows = [
        _row(time_period="2019", value="10"),
        _row(time_period="2020", value="20"),
    ]

    aggregates = aggregate_observations_by_facet(rows)

    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate == SdmxFacetAggregate(
        facet_id="facet1",
        provenance="OECD_DAC2A",
        unit="USD",
        measurement_method="GrantEquivalent",
        scaling_factor="0",
        periods=("2019", "2020"),
        row_count=2,
    )


def test_distinct_facet_ids_aggregate_separately_in_first_seen_order():
    rows = [
        _row(facet_id="facet2", time_period="2018"),
        _row(facet_id="facet1", time_period="2020"),
        _row(facet_id="facet2", time_period="2019"),
    ]

    aggregates = aggregate_observations_by_facet(rows)

    assert [a.facet_id for a in aggregates] == ["facet2", "facet1"]
    assert aggregates[0].periods == ("2018", "2019")
    assert aggregates[0].row_count == 2
    assert aggregates[1].periods == ("2020",)
    assert aggregates[1].row_count == 1


def test_periods_sort_ascending_regardless_of_row_order():
    rows = [_row(time_period="2022"), _row(time_period="2019"), _row(time_period="2020")]

    (aggregate,) = aggregate_observations_by_facet(rows)

    assert aggregate.periods == ("2019", "2020", "2022")


def test_row_with_no_time_period_is_skipped_from_periods_but_counted():
    rows = [_row(time_period="2020"), _row(time_period=None)]

    (aggregate,) = aggregate_observations_by_facet(rows)

    assert aggregate.periods == ("2020",)
    assert aggregate.row_count == 2


def test_rows_with_no_facet_id_group_together():
    rows = [_row(facet_id=None, time_period="2020"), _row(facet_id=None, time_period="2021")]

    (aggregate,) = aggregate_observations_by_facet(rows)

    assert aggregate.facet_id is None
    assert aggregate.row_count == 2


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("unit", "EUR"),
        ("measurement_method", "Amount"),
        ("provenance", "IMF_WEO"),
        ("scaling_factor", "3"),
    ],
)
def test_disagreement_within_one_facet_id_raises(field_name: str, value: str):
    rows = [_row(), _row(**{field_name: value})]

    with pytest.raises(SdmxResponseError) as excinfo:
        aggregate_observations_by_facet(rows)

    assert "facet1" in str(excinfo.value)
    assert field_name in str(excinfo.value)


def test_empty_rows_returns_empty_list():
    assert aggregate_observations_by_facet([]) == []
