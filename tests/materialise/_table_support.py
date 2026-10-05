"""Build observation-contract tables from shared test rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa

from dc_sdmx.results import (
    ATTRIBUTE_COLUMNS,
    DIMENSION_COLUMNS,
    VALUE_COLUMN,
    VARIABLE_COLUMN,
    SdmxObservation,
    observation_columns,
    observation_schema,
)


def table_from_cells(rows: Iterable[Mapping[str, str | None]], *, roles: Sequence[str]) -> pa.Table:
    """Build a table from wire-keyed cells; missing cells are null except
    missing `OBS_VALUE`, which is `""`.
    """
    columns = observation_columns(roles)
    ordered_rows = list(rows)
    data: dict[str, list[str | None]] = {}
    for name in columns:
        default = "" if name == VALUE_COLUMN else None
        data[name] = [row.get(name, default) for row in ordered_rows]
    return pa.Table.from_pydict(data, schema=observation_schema(roles))


def table_from_observations(rows: Iterable[SdmxObservation], *, roles: Sequence[str]) -> pa.Table:
    """Build an observation-contract table from parsed SDMX rows."""
    ordered_rows = list(rows)
    field_by_dimension = {
        "unit": "unit",
        "measurementMethod": "measurement_method",
        "observationPeriod": "observation_period",
        "provenance": "provenance",
        "TIME_PERIOD": "time_period",
    }
    field_by_attribute = {"scalingFactor": "scaling_factor", "facetId": "facet_id"}

    data: dict[str, list[str | None]] = {VARIABLE_COLUMN: [row.variable for row in ordered_rows]}
    for role in roles:
        data[role] = [row.role_values.get(role) for row in ordered_rows]
    for name in DIMENSION_COLUMNS:
        field = field_by_dimension[name]
        data[name] = [getattr(row, field) for row in ordered_rows]
    data[VALUE_COLUMN] = [row.value for row in ordered_rows]
    for name in ATTRIBUTE_COLUMNS:
        field = field_by_attribute[name]
        data[name] = [getattr(row, field) for row in ordered_rows]

    return pa.Table.from_pydict(data, schema=observation_schema(roles))
