"""Typed results and parsers for SDMX-CSV 2.0.0 and SDMX-JSON structure 2.0.0.

Pure parsers accept fetched data. Lazy row and Arrow-batch readers support
streaming with bounded reads. Result types are frozen Pydantic dataclasses.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import IO, TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pydantic

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

from dc_sdmx.errors import SdmxResponseError

# Missing dimension cells use this sentinel. Measure and attribute cells use `""`.
_NOT_APPLICABLE = "NotApplicable"

# Headers contain this prefix, dynamic role columns, then named trailing columns.
# Validation uses column names because deployments may omit trailing `facetId`.
_HEADER_PREFIX = ("STRUCTURE", "STRUCTURE_ID", "ACTION", "variableMeasured")
_REQUIRED_TRAILING_COLUMNS = (
    "unit",
    "measurementMethod",
    "observationPeriod",
    "provenance",
    "TIME_PERIOD",
    "OBS_VALUE",
    "scalingFactor",
)
_OPTIONAL_TRAILING_COLUMNS = ("facetId",)
_TRAILING_COLUMNS = frozenset(_REQUIRED_TRAILING_COLUMNS + _OPTIONAL_TRAILING_COLUMNS)

# Readers and consumers share this all-string observation table contract.

# The observed variable's bare-form dcid. Always present.
VARIABLE_COLUMN = "variableMeasured"

# Dimension and role columns map `NotApplicable` or an absent column to null. An empty role
# cell names no entity and maps to null too; other empty cells remain `""`. Role columns vary
# by response.
DIMENSION_COLUMNS = ("unit", "measurementMethod", "observationPeriod", "provenance", "TIME_PERIOD")

# Raw `OBS_VALUE` string, empty when the row has no observation.
VALUE_COLUMN = "OBS_VALUE"

# Empty or absent attribute cells map to null.
ATTRIBUTE_COLUMNS = ("scalingFactor", "facetId")


def observation_columns(roles: Sequence[str]) -> tuple[str, ...]:
    """Return `variableMeasured`, roles, and trailing columns in wire order."""
    return (VARIABLE_COLUMN, *roles, *_REQUIRED_TRAILING_COLUMNS, *_OPTIONAL_TRAILING_COLUMNS)


def observation_schema(roles: Sequence[str]) -> pa.Schema:
    """Return the all-string schema for `observation_columns(roles)`."""
    return pa.schema([(name, pa.string()) for name in observation_columns(roles)])


def observed_value(cell: str | None) -> float | None:
    """The observation an `OBS_VALUE` cell holds: the float it parses as, or `None` for an
    absent, empty or unparseable cell, which holds none. NaN and the infinities parse, so
    they are observations. Every reader of a value takes this rule: a row parsed here
    (`SdmxObservation.value_as_float`), a column of cells
    (`materialise.columnar.observed_values`) and a parsed point (`is_observed`)."""
    if not cell:
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def is_observed(value: float | None) -> bool:
    """Whether `value`, a cell parsed by `observed_value`, holds an observation."""
    return value is not None


@pydantic.dataclasses.dataclass(frozen=True)
class SdmxObservation:
    """One SDMX-CSV data row.

    NA or empty entity-role cells are omitted from `role_values`. Dimension fields are
    `None` for the wire's NA sentinel or an absent column. Attribute fields
    are `None` for empty cells or absent columns. `value` keeps raw `OBS_VALUE`.
    `value_as_float()` parses it when possible.
    """

    variable: str
    role_values: Mapping[str, str]
    unit: str | None
    measurement_method: str | None
    observation_period: str | None
    provenance: str | None
    time_period: str | None
    value: str
    scaling_factor: str | None
    facet_id: str | None

    def value_as_float(self) -> float | None:
        """The observation `value` holds (`observed_value`), or `None` if it holds none."""
        return observed_value(self.value)


@pydantic.dataclasses.dataclass(frozen=True)
class SdmxFacetAggregate:
    """Aggregate for one `facet_id`, independent of any consumer's facet type.

    `periods` contains sorted, non-empty row periods. `row_count` includes
    rows without a period. For a `LATEST` read, each period is one entity's
    latest point and `row_count` counts series. Full-history reads count observation rows.
    """

    facet_id: str | None
    provenance: str | None
    unit: str | None
    measurement_method: str | None
    scaling_factor: str | None
    periods: tuple[str, ...]
    row_count: int


def aggregate_observations_by_facet(
    rows: Sequence[SdmxObservation],
) -> list[SdmxFacetAggregate]:
    """Group rows by `facet_id` in first-seen order. Rows without an id share a group.

    Each group must agree on unit, method, provenance, and scaling factor or
    `_check_facet_consistency` raises `SdmxResponseError`.
    """
    by_facet: dict[str | None, list[SdmxObservation]] = {}
    for row in rows:
        by_facet.setdefault(row.facet_id, []).append(row)

    aggregates: list[SdmxFacetAggregate] = []
    for facet_id, group in by_facet.items():
        _check_facet_consistency(facet_id, group)
        periods = tuple(sorted(row.time_period for row in group if row.time_period))
        aggregates.append(
            SdmxFacetAggregate(
                facet_id=facet_id,
                provenance=group[0].provenance,
                unit=group[0].unit,
                measurement_method=group[0].measurement_method,
                scaling_factor=group[0].scaling_factor,
                periods=periods,
                row_count=len(group),
            )
        )
    return aggregates


def _check_facet_consistency(facet_id: str | None, rows: list[SdmxObservation]) -> None:
    """Raise `SdmxResponseError` if rows sharing a facet disagree on series metadata."""
    for field_name in ("unit", "measurement_method", "provenance", "scaling_factor"):
        values = {getattr(row, field_name) for row in rows}
        if len(values) > 1:
            raise SdmxResponseError(
                f"SDMX rows sharing facet_id {facet_id!r} disagree on "
                f"{field_name!r}: {sorted(v for v in values if v is not None)!r}"
            )


@pydantic.dataclasses.dataclass(frozen=True)
class SdmxDataResult:
    """A parsed `/data` response with roles in header order and rows in response order."""

    roles: tuple[str, ...]
    rows: tuple[SdmxObservation, ...]


@pydantic.dataclasses.dataclass(frozen=True)
class SdmxAvailabilityResult:
    """A parsed `/availability` response.

    `components` maps each component id to the first-seen, deduplicated union
    of member values. No cube regions yields an empty mapping. `raw` holds the
    decoded body for one request and `{}` for merged chunks, which have no
    single response body.
    """

    components: Mapping[str, tuple[str, ...]]
    raw: dict


def parse_data_csv(text: str) -> SdmxDataResult:
    """Parse a full SDMX-CSV `/data` body eagerly into `SdmxDataResult`.

    Raises `SdmxResponseError` for a missing or invalid header, an unknown or
    missing required column, or a row-width mismatch. Direct parser errors
    have `upstream_status=None`. `SdmxClient.data()` adds the HTTP status.
    """
    roles, rows = iter_data_csv_rows(io.StringIO(text))
    return SdmxDataResult(roles=roles, rows=tuple(rows))


def iter_data_csv_rows(lines: Iterable[str]) -> tuple[tuple[str, ...], Iterator[SdmxObservation]]:
    """Validate the header eagerly and return a lazy iterator over parsed rows.

    `lines` must preserve terminators so `csv.reader` can rejoin newlines in
    quoted fields. `Response.iter_lines()` strips them. Lazy parsing lets a
    caller stop at a row cap without parsing the rest of the body.
    """
    reader = csv.reader(lines)
    try:
        header = next(reader)
    except StopIteration:
        raise SdmxResponseError("SDMX-CSV response has no header row") from None

    roles = resolve_data_header(header)

    def _rows() -> Iterator[SdmxObservation]:
        for i, raw_row in enumerate(reader, start=1):
            yield _parse_row(header, roles, raw_row, row_index=i)

    return roles, _rows()


def resolve_data_header(header: Sequence[str]) -> tuple[str, ...]:
    """Validate the SDMX-CSV header and return entity-role columns in header order.

    Raises `SdmxResponseError` for a missing fixed prefix, unknown trailing
    column, or missing required trailing column. `facetId` is optional.
    """
    if tuple(header[: len(_HEADER_PREFIX)]) != _HEADER_PREFIX:
        raise SdmxResponseError(f"unexpected SDMX-CSV header: {header!r}")

    rest = header[len(_HEADER_PREFIX) :]
    boundary = next((i for i, name in enumerate(rest) if name in _TRAILING_COLUMNS), len(rest))
    roles = tuple(rest[:boundary])
    trailing = rest[boundary:]

    unknown = [name for name in trailing if name not in _TRAILING_COLUMNS]
    if unknown:
        raise SdmxResponseError(
            f"unrecognized SDMX-CSV trailing column(s) {unknown!r} in {header!r}"
        )
    missing = [name for name in _REQUIRED_TRAILING_COLUMNS if name not in trailing]
    if missing:
        raise SdmxResponseError(
            f"SDMX-CSV header is missing required column(s) {missing!r}: {header!r}"
        )
    return roles


# PyArrow's `ArrowInvalid` marker for malformed UTF-8 in a CSV cell.
_MALFORMED_UTF8_MARKER = "invalid UTF8"

# `open_csv()` uses this marker for an empty stream after the header.
_EMPTY_CSV_MARKER = "Empty CSV file"

_NOT_APPLICABLE_SCALAR = pa.scalar(_NOT_APPLICABLE, type=pa.string())
_EMPTY_STRING_SCALAR = pa.scalar("", type=pa.string())
_NULL_STRING_SCALAR = pa.scalar(None, type=pa.string())


class _HeaderPeekReader(io.RawIOBase):
    """Read and validate the header before pyarrow consumes the remaining byte stream.

    `peek_header()` reads through the first newline and buffers any following
    bytes. `readinto()` replays those bytes before reading from `source`.
    """

    def __init__(self, source: io.BufferedReader) -> None:
        self._source = source
        self._replay = b""

    def readable(self) -> bool:
        return True

    def peek_header(self) -> bytes:
        buf = bytearray()
        chunk = bytearray(io.DEFAULT_BUFFER_SIZE)
        while b"\n" not in buf:
            n = self._source.readinto(chunk)
            if not n:
                break
            buf += chunk[:n]
        newline_index = buf.find(b"\n")
        if newline_index == -1:
            self._replay = b""
            return bytes(buf)
        header_line = bytes(buf[: newline_index + 1])
        self._replay = bytes(buf[newline_index + 1 :])
        return header_line

    def readinto(self, buffer: WriteableBuffer, /) -> int:
        if self._replay:
            view = memoryview(buffer)
            chunk, self._replay = self._replay[: len(view)], self._replay[len(view) :]
            view[: len(chunk)] = chunk
            return len(chunk)
        return self._source.readinto(buffer)


def iter_data_csv_batches(
    source: io.RawIOBase, *, block_size: int = 1 << 20
) -> tuple[tuple[str, ...], Iterator[pa.RecordBatch]]:
    """Validate the header and return a lazy Arrow iterator under `observation_schema(roles)`.

    Wrap `source` in `io.BufferedReader` before passing it to pyarrow. PyArrow
    accepts short reads as whole blocks without retrying. Buffering ensures
    each read fills `block_size`, limits pyarrow's read-ahead buffer, and
    prevents ordinary rows from crossing its two-block limit. `_HeaderPeekReader`
    validates the header first. Empty
    input raises `SdmxResponseError`. A header-only body yields no batches.

    Reads use `use_threads=False`, `newlines_in_values=True`, and string
    columns with null conversion disabled. Batches drop the three structure
    columns, map role and dimension `NotApplicable`, empty roles and empty attributes to null,
    preserve `OBS_VALUE`, and fill an absent optional `facetId` with nulls.

    Malformed UTF-8, empty rows, and column-count mismatches raise
    `SdmxResponseError`. PyArrow's batch-level mismatch message identifies
    the batch after the number of rows already yielded. Other source errors,
    including byte-cap errors, propagate unchanged.
    """
    wrapper = _HeaderPeekReader(io.BufferedReader(source, buffer_size=block_size))
    header_line = wrapper.peek_header()
    if not header_line:
        raise SdmxResponseError("SDMX-CSV response has no header row")
    try:
        (header,) = csv.reader([header_line.decode("utf-8")])
    except UnicodeDecodeError as exc:
        raise SdmxResponseError(
            f"SDMX /data response body contained malformed UTF-8: {exc}"
        ) from exc
    roles = resolve_data_header(header)

    try:
        reader = pacsv.open_csv(
            # PyArrow reads any source with `readinto`.
            # Its stub requires an `IO` cast.
            cast(IO[bytes], wrapper),
            read_options=pacsv.ReadOptions(
                column_names=header, use_threads=False, block_size=block_size
            ),
            parse_options=pacsv.ParseOptions(
                newlines_in_values=True,
                # Retain blank records for explicit validation.
                ignore_empty_lines=False,
            ),
            convert_options=pacsv.ConvertOptions(
                column_types={name: pa.string() for name in header},
                strings_can_be_null=False,
            ),
        )
    except pa.ArrowInvalid as exc:
        # A header-only body has no bytes for `open_csv()` to buffer and yields zero batches.
        if _EMPTY_CSV_MARKER in str(exc):
            return roles, iter(())
        # Initial read-ahead can raise here. Later malformed rows raise from `read_next_batch()`.
        raise SdmxResponseError(_batch_error_message(str(exc), rows_yielded=0)) from exc

    def _batches() -> Iterator[pa.RecordBatch]:
        rows_yielded = 0
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                return
            except pa.ArrowInvalid as exc:
                raise SdmxResponseError(
                    _batch_error_message(str(exc), rows_yielded=rows_yielded)
                ) from exc
            _reject_empty_records(batch, rows_yielded=rows_yielded)
            rows_yielded += batch.num_rows
            yield _contract_batch(batch, roles)

    return roles, _batches()


def _reject_empty_records(batch: pa.RecordBatch, *, rows_yielded: int) -> None:
    """Reject rows whose raw CSV fields are all empty strings."""
    empty = pc.equal(batch.column(0), _EMPTY_STRING_SCALAR)
    for column in batch.columns[1:]:
        matches = pc.equal(column, _EMPTY_STRING_SCALAR)
        empty = pc.and_(empty, matches)
    if pc.any(empty).as_py():
        offset = pc.index(empty, True).as_py()
        assert offset is not None
        row_index = rows_yielded + offset + 1
        raise SdmxResponseError(f"SDMX-CSV data row {row_index} is empty")


def _batch_error_message(message: str, *, rows_yielded: int) -> str:
    """Map Arrow errors to SDMX parser messages.

    Malformed UTF-8 has its own message. Column-count errors name the batch
    after `rows_yielded` because PyArrow supplies no row number.
    """
    if _MALFORMED_UTF8_MARKER in message:
        return f"SDMX /data response body contained malformed UTF-8: {message}"
    return (
        "SDMX /data response body has a malformed row in the batch after "
        f"data row {rows_yielded}: {message}"
    )


def _contract_batch(batch: pa.RecordBatch, roles: tuple[str, ...]) -> pa.RecordBatch:
    """Rebuild one raw pyarrow CSV batch onto `observation_schema(roles)`,
    applying the table contract's null mapping (see `iter_data_csv_batches`).
    """
    schema = observation_schema(roles)
    sentinel_mapped = frozenset((*roles, *DIMENSION_COLUMNS))
    empty_mapped = frozenset((*roles, *ATTRIBUTE_COLUMNS))
    names = frozenset(batch.schema.names)
    arrays = []
    for name in schema.names:
        if name not in names:
            arrays.append(pa.nulls(batch.num_rows, type=pa.string()))
            continue
        column = batch.column(name)
        if name in sentinel_mapped:
            column = pc.if_else(
                pc.equal(column, _NOT_APPLICABLE_SCALAR),
                _NULL_STRING_SCALAR,
                column,
            )
        if name in empty_mapped:
            column = pc.if_else(
                pc.equal(column, _EMPTY_STRING_SCALAR),
                _NULL_STRING_SCALAR,
                column,
            )
        arrays.append(column)
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _parse_row(
    header: list[str], roles: tuple[str, ...], raw_row: list[str], *, row_index: int
) -> SdmxObservation:
    try:
        row = dict(zip(header, raw_row, strict=True))
    except ValueError:
        raise SdmxResponseError(
            f"SDMX-CSV data row {row_index} has {len(raw_row)} column(s), expected {len(header)}"
        ) from None
    # A `NotApplicable` or empty role cell names no entity, so the role is absent.
    role_values = {role: row[role] for role in roles if row[role] not in (_NOT_APPLICABLE, "")}
    # Dimensions map NA or absent columns to `None`.
    # Attributes map empty or absent cells to `None`.
    dimensions = {name: _none_if_na(row.get(name)) for name in DIMENSION_COLUMNS}
    attributes = {name: row.get(name) or None for name in ATTRIBUTE_COLUMNS}
    return SdmxObservation(
        variable=row[VARIABLE_COLUMN],
        role_values=role_values,
        unit=dimensions["unit"],
        measurement_method=dimensions["measurementMethod"],
        observation_period=dimensions["observationPeriod"],
        provenance=dimensions["provenance"],
        time_period=dimensions["TIME_PERIOD"],
        value=row.get(VALUE_COLUMN) or "",
        scaling_factor=attributes["scalingFactor"],
        facet_id=attributes["facetId"],
    )


def _none_if_na(value: str | None) -> str | None:
    return None if value is None or value == _NOT_APPLICABLE else value


def _as_list(value: object, *, field: str) -> list:
    """Require a JSON array. Explicit `null` and other present non-arrays are invalid."""
    if not isinstance(value, list):
        raise SdmxResponseError(
            f"SDMX availability response {field!r} is not a JSON array: {value!r}"
        )
    return value


def _as_dict(value: object, *, field: str) -> dict:
    """Validate one array ITEM (a constraint/region/keyValue entry) is a JSON object."""
    if not isinstance(value, dict):
        raise SdmxResponseError(
            f"SDMX availability response {field!r} entry is not a JSON object: {value!r}"
        )
    return value


def _require_key(container: dict, key: str, *, field: str) -> object:
    """Require non-optional SDMX-JSON keys `data` and `dataConstraints`.

    Their absence means a malformed response, even when no data is available.
    Deeper `omitempty` collections may be absent and default to empty lists.
    """
    if key not in container:
        raise SdmxResponseError(f"SDMX availability response is missing required field {field!r}")
    return container[key]


def parse_availability_json(body: dict) -> SdmxAvailabilityResult:
    """Parse an SDMX-JSON structure 2.0.0 `/availability` body.

    Raises `SdmxResponseError` unless the body, required `data` and
    `dataConstraints` keys, present collections, and array items match the
    schema. Missing deeper `omitempty` collections mean empty results.
    Exclusion entries (`include: false`) are unsupported. Absent `include`
    defaults to true. Empty or missing component ids are skipped, while
    truthy non-string ids and invalid member values raise.

    Direct parser errors have `upstream_status=None`. The client adds the
    response status.
    """
    if not isinstance(body, dict):
        raise SdmxResponseError(f"SDMX availability response body is not a JSON object: {body!r}")
    data = _require_key(body, "data", field="data")
    if not isinstance(data, dict):
        raise SdmxResponseError(f"SDMX availability response 'data' is not a JSON object: {data!r}")

    components: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    data_constraints_raw = _require_key(data, "dataConstraints", field="data.dataConstraints")
    constraints = _as_list(data_constraints_raw, field="data.dataConstraints")
    for raw_constraint in constraints:
        constraint = _as_dict(raw_constraint, field="dataConstraints[]")
        regions = _as_list(constraint.get("cubeRegions", []), field="cubeRegions")
        for raw_region in regions:
            region = _as_dict(raw_region, field="cubeRegions[]")
            # Reuse the validated list when reporting excluded component IDs.
            key_values = _as_list(region.get("keyValues", []), field="keyValues")
            if region.get("include", True) is False:
                excluded_ids = [kv.get("id") for kv in key_values if isinstance(kv, dict)]
                raise SdmxResponseError(
                    "SDMX availability response cube region has include=false "
                    f"(component(s): {excluded_ids!r}); exclusion semantics are not "
                    "supported by this client"
                )
            for raw_key_value in key_values:
                key_value = _as_dict(raw_key_value, field="keyValues[]")
                component_id = key_value.get("id")
                if not component_id:
                    continue
                if not isinstance(component_id, str):
                    raise SdmxResponseError(
                        f"SDMX availability response keyValues[].id is not a string: "
                        f"{component_id!r}"
                    )
                if key_value.get("include", True) is False:
                    raise SdmxResponseError(
                        f"SDMX availability response component {component_id!r} has "
                        "include=false; exclusion semantics are not supported by this client"
                    )
                bucket = components.setdefault(component_id, [])
                bucket_seen = seen.setdefault(component_id, set())
                values = _as_list(key_value.get("values", []), field="keyValues[].values")
                for raw_value in values:
                    value = raw_value.get("value") if isinstance(raw_value, dict) else raw_value
                    if not isinstance(value, str):
                        raise SdmxResponseError(
                            f"SDMX availability response component {component_id!r} has "
                            f"a non-string value: {raw_value!r}"
                        )
                    if value not in bucket_seen:
                        bucket_seen.add(value)
                        bucket.append(value)
    return SdmxAvailabilityResult(
        components={component_id: tuple(values) for component_id, values in components.items()},
        raw=body,
    )
