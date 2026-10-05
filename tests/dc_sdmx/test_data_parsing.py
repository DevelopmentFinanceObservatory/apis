"""Parse SDMX-CSV fixtures matching the mixer's formatter cases."""

from __future__ import annotations

import pytest

from dc_sdmx.errors import SdmxResponseError
from dc_sdmx.results import parse_data_csv

from .conftest import read_fixture_text


def test_header_only_has_no_rows():
    result = parse_data_csv(read_fixture_text("data_header_only.csv"))
    assert result.roles == ("observationAbout",)
    assert result.rows == ()


def test_single_entity_rows_preserve_order_and_values():
    result = parse_data_csv(read_fixture_text("data_single_entity.csv"))
    assert result.roles == ("observationAbout",)
    assert len(result.rows) == 2

    first, second = result.rows
    assert first.variable == "Count_Person"
    assert first.role_values == {"observationAbout": "country/USA"}
    assert first.unit == "Person"
    assert first.measurement_method == "Census"
    assert first.observation_period == "P1Y"
    assert first.provenance == "dc/base"
    assert first.time_period == "2020"
    assert first.value == "1.50"
    assert first.value_as_float() == pytest.approx(1.50)
    assert first.scaling_factor == "0"
    assert first.facet_id == "stored-facet-id"

    assert second.time_period == "2021"
    assert second.value == "2"


def test_na_fallback_maps_dimension_cells_to_none():
    result = parse_data_csv(read_fixture_text("data_single_entity_na.csv"))
    populated, empty = result.rows

    # Dimension columns map the wire's `NotApplicable` sentinel to `None`.
    assert populated.unit is None
    assert populated.measurement_method is None
    assert populated.observation_period is None
    assert populated.provenance is None
    assert populated.time_period == "2020"
    assert populated.scaling_factor == "0"
    assert populated.facet_id == "stored-facet-id"

    # `NotApplicable` maps TIME_PERIOD to None. Empty OBS_VALUE stays `""`;
    # empty scaling and facet cells map to None.
    assert empty.time_period is None
    assert empty.value == ""
    assert empty.value_as_float() is None
    assert empty.scaling_factor is None
    assert empty.facet_id is None
    # The role itself is bound -- only the observation-scoped fields are absent.
    assert empty.role_values == {"observationAbout": "country/USA"}


def test_multi_entity_header_and_role_mapping():
    result = parse_data_csv(read_fixture_text("data_multi_entity.csv"))
    assert result.roles == ("destinationCountry", "sourceCountry")
    assert "observationAbout" not in result.roles

    (row,) = result.rows
    assert row.role_values == {
        "destinationCountry": "country/CAN",
        "sourceCountry": "country/USA",
    }
    assert row.unit is None
    assert row.measurement_method is None
    assert row.observation_period is None
    assert row.provenance == "dc/base"
    assert row.scaling_factor is None
    assert row.facet_id is None


def test_csv_escaping_round_trips_quotes_and_commas():
    result = parse_data_csv(read_fixture_text("data_escaping.csv"))
    (row,) = result.rows
    assert row.variable == 'Count,"Person"'
    assert row.value == "foo,bar"


def test_malformed_header_raises_response_error():
    with pytest.raises(SdmxResponseError, match="unexpected SDMX-CSV header") as excinfo:
        parse_data_csv("not,a,valid,header\r\n")
    # upstream_status is None -- this call has no live response to source it from.
    assert excinfo.value.upstream_status is None


def test_empty_body_raises_response_error():
    with pytest.raises(SdmxResponseError, match="no header row"):
        parse_data_csv("")


def test_unrecognized_trailing_column_raises_response_error():
    # A role name would never legitimately collide with a known trailing
    # column, but an unrecognized name appearing where a trailing column is
    # expected (after the first known trailing column) can't be interpreted.
    with pytest.raises(SdmxResponseError, match="unrecognized SDMX-CSV trailing column"):
        parse_data_csv(
            "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,bogus\r\n"
        )


def test_header_missing_a_required_trailing_column_raises_response_error():
    # Only `facetId` is optional; missing another trailing column misaligns rows.
    with pytest.raises(SdmxResponseError, match="missing required column"):
        parse_data_csv(
            "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,OBS_VALUE\r\n"
        )


_SINGLE_ENTITY_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    "scalingFactor,facetId\r\n"
)


def test_truncated_row_raises_response_error_naming_row_and_counts():
    # 10 cells cannot map safely to the 13-column header.
    truncated_row = (
        "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,Person,Census,P1Y,dc/base,2020\r\n"
    )
    with pytest.raises(SdmxResponseError, match=r"row 1 has 10 column\(s\), expected 13"):
        parse_data_csv(_SINGLE_ENTITY_HEADER + truncated_row)


def test_extra_columns_row_raises_response_error_naming_row_and_counts():
    extra_columns_row = (
        "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,Person,Census,P1Y,"
        "dc/base,2020,1.50,0,stored-facet-id,EXTRA\r\n"
    )
    with pytest.raises(SdmxResponseError, match=r"row 1 has 14 column\(s\), expected 13"):
        parse_data_csv(_SINGLE_ENTITY_HEADER + extra_columns_row)


def test_malformed_row_index_counts_only_data_rows():
    # Row numbers start at 1 after the header.
    good_row = "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA,Person,Census,P1Y,dc/base,2020,1.50,0,stored-facet-id\r\n"
    truncated_row = "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/USA\r\n"
    with pytest.raises(SdmxResponseError, match=r"row 2 has 5 column\(s\), expected 13"):
        parse_data_csv(_SINGLE_ENTITY_HEADER + good_row + truncated_row)


def test_header_missing_facet_id_column_omits_it_as_none():
    # Some deployments omit `facetId`; name-based parsing leaves `facet_id` null.
    result = parse_data_csv(read_fixture_text("data_no_facet_id_column.csv"))
    assert result.roles == ("observationAbout",)
    (row,) = result.rows
    assert row.variable == "Count_Person"
    assert row.unit == "Person"
    assert row.scaling_factor == "0"
    assert row.facet_id is None
