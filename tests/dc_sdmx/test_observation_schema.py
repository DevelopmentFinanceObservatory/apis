"""Tests for the observation table contract: `observation_columns`,
`observation_schema`, and the `tests.materialise._table_support` builders.
"""

from __future__ import annotations

import pyarrow as pa

from dc_sdmx.results import iter_data_csv_rows, observation_columns, observation_schema
from tests.materialise._table_support import table_from_cells, table_from_observations


def test_observation_columns_with_no_roles():
    assert observation_columns(()) == (
        "variableMeasured",
        "unit",
        "measurementMethod",
        "observationPeriod",
        "provenance",
        "TIME_PERIOD",
        "OBS_VALUE",
        "scalingFactor",
        "facetId",
    )


def test_observation_columns_with_one_role():
    assert observation_columns(("observationAbout",)) == (
        "variableMeasured",
        "observationAbout",
        "unit",
        "measurementMethod",
        "observationPeriod",
        "provenance",
        "TIME_PERIOD",
        "OBS_VALUE",
        "scalingFactor",
        "facetId",
    )


def test_observation_columns_keeps_role_order_with_two_roles():
    # Roles land between `variableMeasured` and the trailing run, in header
    # order.
    assert observation_columns(("destinationCountry", "sourceCountry")) == (
        "variableMeasured",
        "destinationCountry",
        "sourceCountry",
        "unit",
        "measurementMethod",
        "observationPeriod",
        "provenance",
        "TIME_PERIOD",
        "OBS_VALUE",
        "scalingFactor",
        "facetId",
    )


def test_observation_schema_is_all_strings_in_column_order():
    for roles in ((), ("observationAbout",), ("destinationCountry", "sourceCountry")):
        schema = observation_schema(roles)
        assert schema.names == list(observation_columns(roles))
        assert all(field.type == pa.string() for field in schema)


def test_table_from_cells_matches_the_contract_schema():
    table = table_from_cells(
        [{"variableMeasured": "Count_Person", "observationAbout": "country/USA"}],
        roles=("observationAbout",),
    )
    assert table.schema == observation_schema(("observationAbout",))


def test_table_from_cells_missing_keys_use_schema_defaults():
    (row,) = table_from_cells(
        [{"variableMeasured": "Count_Person"}], roles=("observationAbout",)
    ).to_pylist()
    assert row["observationAbout"] is None
    assert row["unit"] is None
    assert row["scalingFactor"] is None
    assert row["facetId"] is None
    assert row["OBS_VALUE"] == ""


# One wire row covers the NA role sentinel, empty OBS_VALUE, and empty facetId rules together.
_BODY_WITH_NA_ROLE_AND_EMPTY_TRAILING_CELLS = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    "scalingFactor,facetId\r\n"
    "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,NotApplicable,Person,Census,"
    "P1Y,dc/base,2020,,0,\r\n"
)


def test_table_from_observations_matches_parsed_rows_under_the_contract():
    roles, rows = iter_data_csv_rows(
        _BODY_WITH_NA_ROLE_AND_EMPTY_TRAILING_CELLS.splitlines(keepends=True)
    )
    parsed_rows = list(rows)
    table = table_from_observations(parsed_rows, roles=roles)

    assert table.schema == observation_schema(roles)
    (row,) = table.to_pylist()
    (parsed,) = parsed_rows

    assert row["variableMeasured"] == parsed.variable == "Count_Person"
    # The NA sentinel leaves the role absent from `role_values` and null in the table.
    assert "observationAbout" not in parsed.role_values
    assert row["observationAbout"] is None
    assert row["unit"] == parsed.unit == "Person"
    assert row["measurementMethod"] == parsed.measurement_method == "Census"
    assert row["observationPeriod"] == parsed.observation_period == "P1Y"
    assert row["provenance"] == parsed.provenance == "dc/base"
    assert row["TIME_PERIOD"] == parsed.time_period == "2020"
    assert parsed.value == ""
    assert row["OBS_VALUE"] == ""
    assert row["scalingFactor"] == parsed.scaling_factor == "0"
    assert parsed.facet_id is None
    assert row["facetId"] is None
