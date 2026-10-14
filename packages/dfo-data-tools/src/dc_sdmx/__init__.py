"""Read-only client for the mixer's `DC/DF_OBS/1.0.0` SDMX endpoints.

`/data` returns SDMX-CSV 2.0.0 observations. `/availability` returns
SDMX-JSON structure 2.0.0. The client exposes row and Arrow-batch readers,
plus typed query, transport, HTTP, response, and cap errors.

The package depends only on `dc_sdmx`, so Search and other consumers can use it
without importing Search-only dependencies.
"""

from __future__ import annotations

from dc_sdmx.client import (
    DEFAULT_PLANNING_LANES,
    ContainedIn,
    EntityFilter,
    SdmxClient,
    place_slices,
    plan_entity_chunks,
    plan_request_chunks,
)
from dc_sdmx.errors import (
    SdmxCapExceededError,
    SdmxClientError,
    SdmxContainmentFilterError,
    SdmxHttpError,
    SdmxIncompatibleVariablesError,
    SdmxMissingFilterError,
    SdmxQueryError,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxTransportError,
    SdmxUnexpectedResponseShapeError,
    SdmxUnsupportedFilterError,
)
from dc_sdmx.results import (
    ATTRIBUTE_COLUMNS,
    DIMENSION_COLUMNS,
    VALUE_COLUMN,
    VARIABLE_COLUMN,
    SdmxAvailabilityResult,
    SdmxDataResult,
    SdmxFacetAggregate,
    SdmxObservation,
    aggregate_observations_by_facet,
    is_observed,
    iter_data_csv_batches,
    iter_data_csv_rows,
    observation_columns,
    observation_schema,
    observed_value,
    resolve_data_header,
)

__all__ = [
    "ATTRIBUTE_COLUMNS",
    "DEFAULT_PLANNING_LANES",
    "DIMENSION_COLUMNS",
    "VALUE_COLUMN",
    "VARIABLE_COLUMN",
    "ContainedIn",
    "EntityFilter",
    "SdmxAvailabilityResult",
    "SdmxCapExceededError",
    "SdmxClient",
    "SdmxClientError",
    "SdmxContainmentFilterError",
    "SdmxDataResult",
    "SdmxFacetAggregate",
    "SdmxHttpError",
    "SdmxIncompatibleVariablesError",
    "SdmxMissingFilterError",
    "SdmxObservation",
    "SdmxQueryError",
    "SdmxResponseError",
    "SdmxRoleCeilingError",
    "SdmxTransportError",
    "SdmxUnexpectedResponseShapeError",
    "SdmxUnsupportedFilterError",
    "aggregate_observations_by_facet",
    "is_observed",
    "iter_data_csv_batches",
    "iter_data_csv_rows",
    "observation_columns",
    "observation_schema",
    "observed_value",
    "place_slices",
    "plan_entity_chunks",
    "plan_request_chunks",
    "resolve_data_header",
]
