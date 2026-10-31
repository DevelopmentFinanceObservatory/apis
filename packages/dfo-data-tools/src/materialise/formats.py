"""Writers for CSV, Excel, Parquet, and the canonical frame, plus Table Schema derivation.

CSV, Excel, and Parquet accept structural `Table` chunks. The first chunk sets column names
and the types of the columns `Rendering.column_types` leaves unset. Later chunks must have the
same columns and are trusted to be in canonical row order. Writers stream rows to a `Sink` and
return metadata whose digest and size cover the bytes they write. Parquet buffers at most one
100,000-row group. Excel uses write-only mode, but its final archive is held in memory so
timestamps can be fixed for stable output.

Headers are sanitised and unique across all three tabular formats and Table Schema. CSV and
Excel escape formula-triggering strings. Excel refuses platform row, column, cell-length,
and XML-character limits. Parquet enforces a byte bound. Reshaped tables use these writers.

`write_canonical_frame` stores grammar `Frame` chunks as the canonical frame: their Arrow table
under `canonical_frame_schema`, in Parquet row groups of a fixed row count, so its bytes depend
on the rows alone. `read_canonical_frame` reads it back as a `Frame`, `read_canonical_frame_table`
reads chosen columns, and `canonical_frame_document` builds the parsed frame1 document from them.
`write_frame1` and `read_frame1_table` write and read the frame1 JSON document that versions
stored before canonical frames hold. `read_stored_frame` reads either by its media type.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import re
import shutil
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq
from dfo_frame_grammar import (
    Frame,
    canonical_frame_schema,
    stream_canonical_frame_bytes,
    unit_cell_to_json,
    unit_cells,
    unit_codes,
)
from dfo_frame_grammar.grammar import (
    FRAME_VERSION,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
    ComposedUnit,
    UnitRequest,
    unit_cell_from_json,
)
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.utils import get_column_letter

from materialise.cell_safety import spreadsheet_safe
from materialise.codes import CodeCatalogue
from materialise.errors import MaterialisationError, MaterialisationUsageError
from materialise.unit_cells import unit_cell_text

if TYPE_CHECKING:
    from typing import IO

    from _typeshed import ReadableBuffer
    from openpyxl.worksheet._write_only import WriteOnlyWorksheet


class Table(Protocol):
    """Structural contract for tabular writers: read-only `columns` and `rows` views."""

    @property
    def columns(self) -> tuple[str, ...]: ...

    @property
    def rows(self) -> tuple[tuple[object, ...], ...]: ...


_ROWS_SLICE = 8192
"""How many rows' worth of Python objects `iter_rows` builds at once for a
`Frame` chunk. Matches `dfo_frame_grammar.frame`'s own streaming slice
bound (`stream_canonical_frame_bytes`'s `_STREAM_SLICE_ROWS`)."""


def iter_rows(table: Table) -> Iterator[tuple[object, ...]]:
    """Yields rows from any `Table`. For `Frame` chunks, reads the Arrow table in
    `_ROWS_SLICE`-row slices to avoid rebuilding all `Frame.rows` at once. Other tables yield
    their stored rows directly."""
    if not isinstance(table, Frame):
        yield from table.rows
        return
    whole = table.table
    for start in range(0, whole.num_rows, _ROWS_SLICE):
        row_slice = whole.slice(start, _ROWS_SLICE)
        yield from Frame(columns=table.columns, table=row_slice).rows


@dataclass(frozen=True)
class Rendering:
    """Per-format options. `None` preserves column names. `friendly_headers` replaces mapped
    names and leaves others unchanged. The mapping applies to CSV, Excel, Parquet, and Table
    Schema, so artifact headers match their schema.

    `column_types` maps a column name to the Table Schema type (`integer`, `number`, `boolean`
    or `string`) the caller fixes for it. CSV, Parquet and Table Schema give that column its
    type whatever its cells hold, so a table with no rows types it as a table with rows does.
    Every other column is typed from the first chunk's cells."""

    friendly_headers: Mapping[str, str] | None = None
    column_types: Mapping[str, str] | None = None


class Sink(Protocol):
    """Write target for CSV, Excel, and Parquet. Writers wrap it with `_HashingWriteWrapper`."""

    def write(self, data: bytes, /) -> int: ...


class _HashingWriteWrapper(io.RawIOBase):
    """Adds incremental hashing, byte counts, and a no-op `flush` for `TextIOWrapper` and
    `ParquetWriter`. `BlobWriter.flush()` otherwise raises `io.UnsupportedOperation`. Forwards
    only `write`. It exposes a write-only, non-seekable stream."""

    # Typeshed requires `.name` for these wrappers. Neither writer reads it.
    name = "<materialise-formats-sink>"

    def __init__(self, target: Sink) -> None:
        super().__init__()
        self._target = target
        self._hasher = hashlib.sha256()
        self._size = 0

    def writable(self) -> bool:
        return True

    def write(self, data: ReadableBuffer, /) -> int:
        self._checkClosed()
        data = bytes(data)
        self._hasher.update(data)
        self._size += len(data)
        self._target.write(data)
        return len(data)

    def tell(self) -> int:
        return self._size

    def flush(self) -> None:
        pass

    def digest(self) -> str:
        return "sha256:" + self._hasher.hexdigest()

    @property
    def size(self) -> int:
        return self._size


@dataclass(frozen=True)
class WrittenArtifact:
    """Metadata for a completed tabular writer call. `digest` is `sha256:<hex>` over the
    exact written bytes, computed incrementally. `size_bytes` counts those bytes. Data stays
    in the supplied `sink`."""

    media_type: str
    digest: str
    size_bytes: int


FRAME1_MEDIA_TYPE = "application/vnd.dfo.frame1+json"
"""Vendor media type returned by `write_frame1`."""

CANONICAL_FRAME_MEDIA_TYPE = "application/vnd.apache.parquet"
"""Media type returned by `write_canonical_frame`. Requested Parquet and series statistics share
it, so a reader finds a version's canonical frame by its rendering."""


# Excel caps are spreadsheet limits. Parquet's byte cap is configurable. CSV and Parquet
# columns are uncapped. `FORMAT_COLUMN_CAPS` exposes column limits without importing writers.
EXCEL_MAX_ROWS = 1_048_576
"""Excel's row ceiling, header row included. A table with this many data
rows already fills the sheet, since the header spends the first row."""

EXCEL_MAX_COLUMNS = 16_384
"""Excel's column ceiling (spreadsheetml XFD, the last column letter)."""

EXCEL_MAX_CELL_CHARS = 32_767
"""Excel's per-cell character limit for string cells."""

PARQUET_MAX_BYTES = 256 * 1024 * 1024
"""Default 256 MiB bound for one Parquet artifact, which is held in memory until written.
This unmeasured fallback can be overridden through `size_bound` when deployment capacity is
known."""

FORMAT_COLUMN_CAPS: Mapping[str, int | None] = {
    "excel": EXCEL_MAX_COLUMNS,
    "csv": None,
    "parquet": None,
}
"""Format -> column cap, `None` meaning uncapped. A reshape engine's pivot
step takes its column cap as a plain argument. This mapping is
the literal data such a caller can mirror, not an import it should take."""


class ExcelLimitError(MaterialisationError):
    """422 `excel_limit`. Raised before workbook creation when row count (including header),
    column count, or a string's character count exceeds Excel limits. Detail gives count and cap."""

    code = "excel_limit"
    status = 422

    def __init__(self, *args: object, detail: str | None = None) -> None:
        super().__init__(*args, detail=detail)


class ExcelIllegalCharacterError(MaterialisationError):
    """422 `excel_illegal_character`. Raised for XML-illegal string cells before they are
    appended. Tab, LF, and CR are allowed. `row` and `column` are zero-based in the sanitised
    header-and-row grid, with row 0 as the header. The offending character remains in the error."""

    code = "excel_illegal_character"
    status = 422

    def __init__(self, *args: object, row: int, column: int, detail: str | None = None) -> None:
        super().__init__(*args, detail=detail)
        self.row = row
        self.column = column


class ParquetSizeBoundError(MaterialisationError):
    """422 `parquet_size_bound`. Raised when encoded bytes exceed `size_bound`, checked after
    each row group and again after close to include footer and schema metadata."""

    code = "parquet_size_bound"
    status = 422

    def __init__(self, *args: object, detail: str | None = None) -> None:
        super().__init__(*args, detail=detail)


FORMATS_CODE_CATALOGUE: CodeCatalogue = {
    "excel_limit": (
        "Excel Limit Exceeded",
        "The table exceeds one of Excel's row, column, or cell-character limits.",
    ),
    "excel_illegal_character": (
        "Excel Illegal Character",
        "A cell contains a character that is illegal in Excel's XML file format.",
    ),
    "parquet_size_bound": (
        "Parquet Size Bound Exceeded",
        "The written Parquet artifact would exceed its configured size bound.",
    ),
}


def _sanitize_csv_cell(value: object) -> object:
    """Adds one apostrophe when a string, after stripping leading apostrophes, starts with
    a formula trigger. The rule is reversible: remove one added apostrophe to recover the
    original value. Leading apostrophes without a following trigger stay unchanged."""
    return spreadsheet_safe(value) if isinstance(value, str) else value


def _check_unique_headers(headers: Sequence[str]) -> None:
    """Raises `MaterialisationUsageError` naming duplicate headers. Duplicates merge columns
    in `_rows_to_arrow_table` and violate Table Schema's unique-field rule."""
    seen: dict[str, int] = {}
    for name in headers:
        seen[name] = seen.get(name, 0) + 1
    duplicates = sorted(name for name, count in seen.items() if count > 1)
    if duplicates:
        raise MaterialisationUsageError(
            f"rendering produced duplicate header name(s) {duplicates!r}; "
            "friendly_headers (or the table's own columns) must map to "
            "distinct names"
        )


def written_header(column: str, *, rendering: Rendering) -> str:
    """The header CSV, Excel, Parquet and Table Schema write for `column`: its friendly name,
    if `rendering` maps one, sanitised like a cell value."""
    labels = rendering.friendly_headers
    return spreadsheet_safe(labels.get(column, column) if labels is not None else column)


def _headers_for(table: Table, *, rendering: Rendering) -> tuple[str, ...]:
    """Each column's `written_header`. CSV, Excel, Parquet, and Table Schema share these
    headers. Duplicate final names raise `MaterialisationUsageError`."""
    headers = tuple(written_header(column, rendering=rendering) for column in table.columns)
    _check_unique_headers(headers)
    return headers


def _prepare_stream(chunks: Iterable[Table]) -> tuple[Table, Iterator[Table]]:
    """Returns the first chunk and a replayable iterator in input order. Headers and types
    come from the first chunk. Later chunks must have identical columns or raise
    `MaterialisationUsageError` with their position. At least one chunk is required, though it
    may be empty. Canonical row order is validated upstream by `dfo_frame_grammar`."""
    iterator = iter(chunks)
    try:
        first = next(iterator)
    except StopIteration:
        raise MaterialisationUsageError(
            "a writer needs at least one chunk (even an empty one) to know its columns"
        ) from None

    def _validated() -> Iterator[Table]:
        yield first
        for index, chunk in enumerate(iterator, start=1):
            if chunk.columns != first.columns:
                raise MaterialisationUsageError(
                    f"chunk {index} has columns {chunk.columns!r}, but the first chunk "
                    f"has columns {first.columns!r}; every chunk in one write must share "
                    "the first chunk's columns"
                )
            yield chunk

    return first, _validated()


def write_csv(chunks: Iterable[Table], *, rendering: Rendering, sink: Sink) -> WrittenArtifact:
    """Streams headers and rows to `sink`, coercing fields from the first chunk and
    sanitising every cell. `None` becomes an empty field. Inline coercion matches Parquet's
    Arrow widening without an intermediate encode/decode pass.

    Uses the default CSV dialect: commas, minimal quoting, and CRLF line endings, stable
    across platforms for the same chunks and rendering."""
    first, stream = _prepare_stream(chunks)
    headers = _headers_for(first, rendering=rendering)
    field_types = _field_types(first, rendering=rendering)

    # `TextIOWrapper` streams UTF-8 without holding both text and bytes copies in memory.
    # `newline=""` preserves the CSV writer's CRLF terminators without platform translation.
    hashing = _HashingWriteWrapper(sink)
    text_wrapper = io.TextIOWrapper(hashing, encoding="utf-8", newline="")
    writer = csv.writer(text_wrapper)
    writer.writerow(headers)
    for chunk in stream:
        for row in iter_rows(chunk):
            writer.writerow(
                [
                    _sanitize_csv_cell(_coerce_cell(value, kind))
                    for value, kind in zip(row, field_types, strict=True)
                ]
            )
    text_wrapper.flush()
    text_wrapper.detach()
    return WrittenArtifact(media_type="text/csv", digest=hashing.digest(), size_bytes=hashing.size)


def check_excel_rows(total_rows: int) -> None:
    """Raise `ExcelLimitError` when `total_rows`, counting the header, exceeds
    `EXCEL_MAX_ROWS`. `write_excel` checks each row as it appends it, so its refusal names the
    first row past the limit."""
    if total_rows > EXCEL_MAX_ROWS:
        raise ExcelLimitError(
            f"{total_rows} rows (including the header) exceeds Excel's "
            f"{EXCEL_MAX_ROWS}-row limit; request csv or parquet instead"
        )


def _check_excel_cell(cell: object, *, row_index: int, column_index: int) -> None:
    """Checks a normalised, sanitised cell and reports zero-based row/column positions
    (header is row 0). Trigger quoting adds a character, so lengths must be checked after
    sanitisation to prevent openpyxl truncating a 32,768-character value. Only strings face
    Excel's character and XML limits."""
    if not isinstance(cell, str):
        return
    if len(cell) > EXCEL_MAX_CELL_CHARS:
        raise ExcelLimitError(
            f"a cell of {len(cell)} characters exceeds Excel's "
            f"{EXCEL_MAX_CELL_CHARS}-character cell limit; request csv or "
            "parquet instead"
        )
    if ILLEGAL_CHARACTERS_RE.search(cell):
        raise ExcelIllegalCharacterError(
            f"the cell at row {row_index}, column {column_index} contains a "
            "character that is illegal in Excel's XML file format; request csv or "
            "parquet instead",
            row=row_index,
            column=column_index,
        )


_FIXED_DOCUMENT_TIMESTAMP = datetime(1980, 1, 1)
"""Fixed workbook creation timestamp for deterministic output."""

_ZIP_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

_MODIFIED_TIMESTAMP_PATTERN = re.compile(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)")
_FIXED_MODIFIED_XML = rb"\g<1>1980-01-01T00:00:00Z\g<2>"


_CORE_XML_NAME = "docProps/core.xml"
_SHEET_PR_END = b"</sheetPr>"

DATA_SHEET_TITLE = "data"
"""The title of an XLSX file's first sheet, which holds the data."""

TOTALS_SHEET_TITLE = "totals"
"""The title of the second sheet a long-layout XLSX file carries when some rows are totals."""

_ZIP_STREAM_CHUNK_BYTES = 1 << 20
"""1 MiB copy buffer for streaming large archive members without holding them in memory."""


def _sheet_xml_name(position: int) -> str:
    """The archive member of the workbook's `position`th sheet, counting from 1."""
    return f"xl/worksheets/sheet{position}.xml"


def _finish_workbook(data: bytes, *, dimensions: Sequence[str]) -> bytes:
    """Fixes ZIP member timestamps and `docProps/core.xml`'s `dcterms:modified`, which
    `openpyxl` stamps with the current time during save. It honours `properties.created`.
    Preserves member order, compression, and attributes.

    Also gives each sheet the `<dimension ref="...">` write-only mode leaves out, `dimensions`
    being each sheet's used range in sheet order, such as `A1:C3`. Write-only mode omits empty
    cells, and without the range a reader that streams rows, as frictionless does through
    openpyxl, reads a row ending in empty cells as short.

    Streams every member except `docProps/core.xml` in `_ZIP_STREAM_CHUNK_BYTES` chunks.
    The core file is a few hundred bytes and is read in full to replace `dcterms:modified`.
    A sheet's first chunk takes the dimension after its `<sheetPr>`, which `openpyxl` writes
    in its first few hundred bytes. Other contents stay bounded to one copy chunk."""
    sheet_dimensions = {
        _sheet_xml_name(position): dimension
        for position, dimension in enumerate(dimensions, start=1)
    }
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            frozen = zipfile.ZipInfo(info.filename, date_time=_ZIP_FIXED_DATE_TIME)
            frozen.compress_type = info.compress_type
            frozen.external_attr = info.external_attr
            if info.filename == _CORE_XML_NAME:
                member_data = src.read(info.filename)
                member_data = _MODIFIED_TIMESTAMP_PATTERN.sub(_FIXED_MODIFIED_XML, member_data)
                dst.writestr(frozen, member_data)
                continue
            with src.open(info.filename) as member_in, dst.open(frozen, mode="w") as member_out:
                if (dimension := sheet_dimensions.get(info.filename)) is not None:
                    head = member_in.read(_ZIP_STREAM_CHUNK_BYTES)
                    member_out.write(_with_dimension(head, dimension, sheet=info.filename))
                shutil.copyfileobj(member_in, member_out, length=_ZIP_STREAM_CHUNK_BYTES)
    return out.getvalue()


def _with_dimension(head: bytes, dimension: str, *, sheet: str) -> bytes:
    """The XML of `sheet`'s first chunk with `<dimension ref="...">` after `</sheetPr>`, its
    place in the schema's element order."""
    end = head.find(_SHEET_PR_END)
    if end < 0:
        raise RuntimeError(f"{sheet} opens without a <sheetPr> to place its dimension")
    end += len(_SHEET_PR_END)
    return head[:end] + f'<dimension ref="{dimension}" />'.encode() + head[end:]


_EXCEL_NATIVE_CELL_TYPES = (str, int, float, bool)
"""Native types accepted by openpyxl. `bool` stays explicit because it subclasses `int` but
must remain a boolean."""


def _cell_text(value: object) -> str:
    """The text a tabular writer puts in a cell that has no native type. A `UnitRequest` or
    `ComposedUnit` cell becomes the JCS serialisation of its `as_json`, the same object frame1
    and the preview carry, for example `{"base_year":2024,"currency":"EUR","prices":"constant"}`.
    Any other value uses `str`."""
    if isinstance(value, (UnitRequest, ComposedUnit)):
        return unit_cell_text(value)
    return str(value)


def _normalize_excel_cell(value: object) -> object:
    """Stringifies values outside `str`/`int`/`float`/`bool`/`None` with `_cell_text` before
    formula sanitisation. This matches CSV and Parquet output. Supported types pass through
    unchanged."""
    if value is None or isinstance(value, _EXCEL_NATIVE_CELL_TYPES):
        return value
    return _cell_text(value)


def write_excel(
    chunks: Iterable[Table],
    *,
    rendering: Rendering,
    sink: Sink,
    totals: Callable[[], Iterable[Table]] | None = None,
) -> WrittenArtifact:
    """Writes literal text and number cells, sanitising formula triggers before openpyxl
    receives strings. Unsupported cell types are stringified first.
    Cell contents are preserved without truncation.

    `chunks` fill the sheet `DATA_SHEET_TITLE`. `totals`, called once that sheet is written,
    gives the rows of a second sheet, `TOTALS_SHEET_TITLE`, with its own header. The sheet is
    left out when `totals` gives no row.

    Write-only mode avoids an in-memory cell graph that measured 2.29 GB at 462,720 rows
    (3.6 times the source data). Checks the column limit before writing, then each sheet's
    row count (including the header), cell length, and XML characters as each row is
    appended. A refusal writes no bytes to `sink`.

    The finished archive stays in memory for timestamp fixing, then goes to `sink` in one
    write. `openpyxl` timestamps ZIP entries and resets `properties.modified` at save time.
    `_finish_workbook` also fixes those fields, while `properties.created` is honoured."""
    first, stream = _prepare_stream(chunks)
    headers = list(_headers_for(first, rendering=rendering))
    n_cols = len(headers)
    if n_cols > EXCEL_MAX_COLUMNS:
        raise ExcelLimitError(
            f"{n_cols} columns exceeds Excel's {EXCEL_MAX_COLUMNS}-column limit; "
            "request csv or parquet instead"
        )

    workbook = Workbook(write_only=True)
    workbook.properties.created = _FIXED_DOCUMENT_TIMESTAMP
    sheets = [workbook.create_sheet(DATA_SHEET_TITLE)]

    # `sheet.append` leaves a write-only XML generator open until save.
    # Close each on refusal before GC can resume it after the output file closes.
    try:
        dimensions = [_append_sheet(sheets[0], headers=headers, chunks=stream)]
        if totals is not None:
            total_chunks = iter(totals())
            if (first_total := next(total_chunks, None)) is not None:
                sheets.append(workbook.create_sheet(TOTALS_SHEET_TITLE))
                dimensions.append(
                    _append_sheet(
                        sheets[1],
                        headers=list(_headers_for(first_total, rendering=rendering)),
                        chunks=itertools.chain((first_total,), total_chunks),
                    )
                )
        buf = io.BytesIO()
        workbook.save(buf)
    except BaseException:
        for sheet in sheets:
            if not sheet.closed:
                sheet.close()
        raise
    data = _finish_workbook(buf.getvalue(), dimensions=dimensions)
    sink.write(data)
    return WrittenArtifact(
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        digest=_digest(data),
        size_bytes=len(data),
    )


def _append_sheet(sheet: WriteOnlyWorksheet, *, headers: list[str], chunks: Iterable[Table]) -> str:
    """Appends `headers` and every row of `chunks` to `sheet`, checking each row and cell
    against Excel's limits, and returns the sheet's used range, such as `A1:C3`."""
    for column_index, cell in enumerate(headers):
        _check_excel_cell(cell, row_index=0, column_index=column_index)
    sheet.append(headers)
    total_rows = 1  # the header row already written

    for chunk in chunks:
        for row in iter_rows(chunk):
            total_rows += 1
            check_excel_rows(total_rows)
            sanitized_row = [_sanitize_csv_cell(_normalize_excel_cell(cell)) for cell in row]
            for column_index, cell in enumerate(sanitized_row):
                _check_excel_cell(cell, row_index=total_rows - 1, column_index=column_index)
            sheet.append(sanitized_row)
    n_cols = len(headers)
    return f"A1:{get_column_letter(n_cols)}{total_rows}" if n_cols else "A1"


def _cell_kind(value: object) -> str:
    """Classifies `bool` separately from `int`, which it subclasses. Unknown types are `any`."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "any"


def _widen_kind(a: str, b: str) -> str:
    """Widens `integer`/`number` to `number`. Any other type mismatch becomes `any`."""
    if a == b:
        return a
    if {a, b} == {"integer", "number"}:
        return "number"
    return "any"


def _infer_column_type(values: Sequence[object]) -> str:
    """Infers a Frictionless type from non-null cells. An all-null column is `any`."""
    kind: str | None = None
    for value in values:
        if value is None:
            continue
        this_kind = _cell_kind(value)
        kind = this_kind if kind is None else _widen_kind(kind, this_kind)
    return kind or "any"


def _field_types(table: Table, *, rendering: Rendering) -> tuple[str, ...]:
    """Each column's type: the one `rendering.column_types` fixes, else `_infer_column_type`
    of its cells."""
    declared = rendering.column_types or {}
    # Read rows once because each `Frame.rows` access rebuilds the full row tuple.
    columns_values: list[list[object]] = [[] for _ in table.columns]
    for row in iter_rows(table):
        for column_values, cell in zip(columns_values, row, strict=True):
            column_values.append(cell)
    return tuple(
        declared.get(column) or _infer_column_type(values)
        for column, values in zip(table.columns, columns_values, strict=True)
    )


def table_schema_for(table: Table, *, rendering: Rendering) -> dict:
    """Returns Table Schema for this table and rendering. Field names match writer headers.
    Types are those `rendering.column_types` fixes, else come from cell values.
    """
    headers = _headers_for(table, rendering=rendering)
    field_types = _field_types(table, rendering=rendering)
    return {
        "fields": [
            {"name": name, "type": field_type}
            for name, field_type in zip(headers, field_types, strict=True)
        ]
    }


_ARROW_TYPES: Mapping[str, pa.DataType] = {
    "boolean": pa.bool_(),
    "integer": pa.int64(),
    "number": pa.float64(),
    "string": pa.string(),
    "any": pa.string(),
}


def _coerce_cell(value: object, kind: str) -> object:
    """Matches Arrow's column coercion without building an array in the CSV writer.
    Stringifies non-string values in `any` columns with `_cell_text` and casts integers to floats
    in `number` columns, excluding `bool`. Nulls and other values pass through unchanged."""
    if value is None:
        return None
    if kind == "any" and not isinstance(value, str):
        return _cell_text(value)
    if kind == "number" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _arrow_schema(headers: Sequence[str], field_types: Sequence[str]) -> pa.Schema:
    return pa.schema(
        [(name, _ARROW_TYPES[kind]) for name, kind in zip(headers, field_types, strict=True)]
    )


def _rows_to_arrow_table(
    headers: Sequence[str],
    field_types: Sequence[str],
    rows: Sequence[Sequence[object]],
    schema: pa.Schema,
) -> pa.Table:
    columns: dict[str, list[object]] = {name: [] for name in headers}
    for row in rows:
        for name, kind, value in zip(headers, field_types, row, strict=True):
            columns[name].append(_coerce_cell(value, kind))
    return pa.Table.from_pydict(columns, schema=schema)


_PARQUET_ROW_GROUP_ROWS = 100_000
"""Sets a 100,000-row group target and flush threshold. A 2.5-million-row file with 7,936
groups of about 317 rows peaked at 1.36 GB on readback as PyArrow prefetched groups.
Coalescing bounds memory to one row group. Streams of 462,720 and 2,517,909 rows flushed
only a small number of times."""


def _write_parquet_stream(
    chunks: Iterable[Table],
    *,
    headers: Sequence[str],
    field_types: Sequence[str],
    schema: pa.Schema,
    size_bound: int | None,
    sink: _HashingWriteWrapper,
) -> None:
    """Coalesces validated rows into `_PARQUET_ROW_GROUP_ROWS` groups, bounding buffered
    memory to one group. Uses headers and types from the first chunk. Checks `size_bound`
    after each flush and after `writer.close()` to include footer bytes. All-empty chunks
    produce zero row groups."""
    # Stubs require nominal `IO`. Runtime accepts this `RawIOBase` write/flush/tell/closed shape.
    writer = pq.ParquetWriter(cast("IO[bytes]", sink), schema)
    buffered_rows: list[Sequence[object]] = []

    def flush() -> None:
        nonlocal buffered_rows
        if not buffered_rows:
            return
        writer.write_table(_rows_to_arrow_table(headers, field_types, buffered_rows, schema))
        buffered_rows = []
        if size_bound is not None and sink.size > size_bound:
            raise ParquetSizeBoundError(
                f"parquet output exceeds its {size_bound}-byte size bound "
                f"({sink.size} bytes written so far); request csv instead"
            )

    for chunk in chunks:
        for row in iter_rows(chunk):
            buffered_rows.append(row)
            if len(buffered_rows) >= _PARQUET_ROW_GROUP_ROWS:
                flush()
    flush()
    writer.close()

    if size_bound is not None and sink.size > size_bound:
        raise ParquetSizeBoundError(
            f"parquet output exceeds its {size_bound}-byte size bound "
            f"({sink.size} bytes total, including the footer); request csv instead"
        )


def write_parquet(
    chunks: Iterable[Table],
    *,
    rendering: Rendering,
    sink: Sink,
    size_bound: int = PARQUET_MAX_BYTES,
) -> WrittenArtifact:
    """Streams Parquet rows into `sink` using first-chunk names and types, matching
    `table_schema_for`. `_write_parquet_stream` enforces row-group and size limits.
    `size_bound` defaults to `PARQUET_MAX_BYTES` and can be overridden. Output is deterministic
    for the same input and PyArrow version because Parquet embeds no wall-clock timestamp."""
    first, stream = _prepare_stream(chunks)
    headers = _headers_for(first, rendering=rendering)
    field_types = _field_types(first, rendering=rendering)
    schema = _arrow_schema(headers, field_types)
    hashing = _HashingWriteWrapper(sink)
    _write_parquet_stream(
        stream,
        headers=headers,
        field_types=field_types,
        schema=schema,
        size_bound=size_bound,
        sink=hashing,
    )
    return WrittenArtifact(
        media_type="application/vnd.apache.parquet",
        digest=hashing.digest(),
        size_bytes=hashing.size,
    )


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


_FRAME1_WRITE_BYTES = 1 << 20
"""Frame1 bytes gathered before one `sink.write`. The encoder yields one piece per row, so
gathering them cuts the per-write cost of hashing and storage while bounding the buffer."""


def write_frame1(chunks: Iterable[Frame], *, sink: Sink) -> WrittenArtifact:
    """Streams the canonical frame1 document of the rows of `chunks`, in order, into `sink`
    and returns `FRAME1_MEDIA_TYPE` metadata. Requires grammar `Frame` chunks in increasing
    `row_sort_key` order. The bytes, digest, and size equal those of one frame holding every
    row. At most `_FRAME1_WRITE_BYTES` of output and one encoder slice of rows are held at
    once. Select/pivot results use tabular writers."""
    hashing = _HashingWriteWrapper(sink)
    pending = bytearray()
    for piece in stream_canonical_frame_bytes(chunks):
        pending += piece
        if len(pending) >= _FRAME1_WRITE_BYTES:
            hashing.write(pending)
            pending.clear()
    if pending:
        hashing.write(pending)
    return WrittenArtifact(
        media_type=FRAME1_MEDIA_TYPE, digest=hashing.digest(), size_bytes=hashing.size
    )


@dataclass(frozen=True)
class Frame1Table:
    """Stored frame1 `columns` and `rows` as tuples, satisfying the `Table` protocol for
    reshaping and tabular writers without rebuilding a grammar `Frame`."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


def read_frame1_table(data: bytes) -> Table:
    """Parses frame1 JSON into a `Frame1Table` for reshaping and tabular writers. Raises
    `ValueError` unless the document is an object with `frame_version`, `columns`, and `rows`,
    the version matches `FRAME_VERSION`, and each row matches the column count.

    Non-null value cells become `float` because whole-number floats parse from JSON as `int`.
    Unit dictionaries are validated and rebuilt as `UnitRequest` or `ComposedUnit` through
    `unit_cell_from_json`, so reshaping keeps hashable, typed cells."""
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ValueError(f"frame1 document must be a JSON object, got {type(document).__name__}")
    missing = [key for key in ("frame_version", "columns", "rows") if key not in document]
    if missing:
        raise ValueError(f"frame1 document is missing key(s) {missing!r}")
    version = document["frame_version"]
    if version != FRAME_VERSION:
        raise ValueError(
            f"frame1 document has frame_version {version!r}, expected {FRAME_VERSION!r}"
        )
    columns = tuple(document["columns"])
    value_index = columns.index(VALUE_COLUMN) if VALUE_COLUMN in columns else None
    unit_index = columns.index(UNIT_COLUMN) if UNIT_COLUMN in columns else None
    rows: list[tuple[object, ...]] = []
    for index, raw_row in enumerate(document["rows"]):
        row = list(raw_row)
        if len(row) != len(columns):
            raise ValueError(
                f"row {index} has {len(row)} cell(s), expected {len(columns)} to match "
                f"the {len(columns)} column(s)"
            )
        if value_index is not None and row[value_index] is not None:
            row[value_index] = float(row[value_index])
        if unit_index is not None and isinstance(row[unit_index], dict):
            row[unit_index] = unit_cell_from_json(row[unit_index])
        rows.append(tuple(row))
    return Frame1Table(columns=columns, rows=tuple(rows))


def frame_entity_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Returns the frame columns between `STAT_VAR_COLUMN` and `TIME_PERIOD_COLUMN`, its
    entity roles. Raises `ValueError` if either marker is missing."""
    columns = tuple(columns)
    if STAT_VAR_COLUMN not in columns:
        raise ValueError(f"table has no {STAT_VAR_COLUMN!r} column")
    if TIME_PERIOD_COLUMN not in columns:
        raise ValueError(f"table has no {TIME_PERIOD_COLUMN!r} column")
    start = columns.index(STAT_VAR_COLUMN) + 1
    end = columns.index(TIME_PERIOD_COLUMN)
    return columns[start:end]


_CANONICAL_FRAME_ROW_GROUP_ROWS = 100_000
"""Rows in each canonical frame row group but the last. A fixed count keeps the bytes independent
of how the rows arrive and bounds the writer's buffer to one group."""

_CANONICAL_FRAME_COMPRESSION = "zstd"


def write_canonical_frame(chunks: Iterable[Frame], *, sink: Sink) -> WrittenArtifact:
    """Streams the canonical frame of the rows of `chunks`, in order, into `sink` and returns
    `CANONICAL_FRAME_MEDIA_TYPE` metadata. Requires grammar `Frame` chunks sharing one column
    tuple, in canonical row order, which `Frame.from_table` trusts on read. The bytes, digest and
    size equal those of one frame holding every row, for one PyArrow version. At most one row
    group of rows is buffered."""
    first, stream = _prepare_stream(chunks)
    schema = canonical_frame_schema(first.columns)
    hashing = _HashingWriteWrapper(sink)
    # Stubs require nominal `IO`. Runtime accepts this `RawIOBase` write/flush/tell/closed shape.
    writer = pq.ParquetWriter(
        cast("IO[bytes]", hashing), schema, compression=_CANONICAL_FRAME_COMPRESSION
    )
    pending: list[pa.Table] = []
    pending_rows = 0
    for chunk in stream:
        pending.append(cast(Frame, chunk).table)
        pending_rows += cast(Frame, chunk).table.num_rows
        while pending_rows >= _CANONICAL_FRAME_ROW_GROUP_ROWS:
            buffered = pa.concat_tables(pending)
            group = buffered.slice(0, _CANONICAL_FRAME_ROW_GROUP_ROWS)
            writer.write_table(group.combine_chunks(), row_group_size=group.num_rows)
            pending = [buffered.slice(_CANONICAL_FRAME_ROW_GROUP_ROWS)]
            pending_rows -= _CANONICAL_FRAME_ROW_GROUP_ROWS
    if pending_rows:
        group = pa.concat_tables(pending)
        writer.write_table(group.combine_chunks(), row_group_size=group.num_rows)
    writer.close()
    return WrittenArtifact(
        media_type=CANONICAL_FRAME_MEDIA_TYPE, digest=hashing.digest(), size_bytes=hashing.size
    )


def read_canonical_frame_table(data: bytes, *, columns: Sequence[str] | None = None) -> pa.Table:
    """The stored canonical frame's table, or only `columns` of it, read from `data` for
    `canonical_frame_rows` and `canonical_frame_document`. Its string columns stay dictionary
    encoded, so each distinct string is decoded once. `pa.Table.validate` checks each dictionary
    index and string before a reader takes them to Python.

    Raises:
        pa.ArrowInvalid: `data` is no Parquet file, or holds an array its own buffers contradict.
    """
    names = list(columns) if columns is not None else pq.read_schema(pa.BufferReader(data)).names
    strings = [name for name in names if name not in _NON_STRING_FRAME_COLUMNS]
    table = pq.read_table(pa.BufferReader(data), columns=names, read_dictionary=strings)
    table.validate(full=True)
    return table


_NON_STRING_FRAME_COLUMNS = frozenset((VALUE_COLUMN, UNIT_COLUMN, STATUS_COLUMN))


def read_canonical_frame(data: bytes) -> Frame:
    """The stored canonical frame `data` holds, every column, as a `Frame`. Raises `FrameError`
    `schema_violation` for a table outside the canonical schema."""
    table = pq.read_table(pa.BufferReader(data))
    return Frame.from_table(columns=tuple(table.column_names), table=table)


def read_stored_frame(data: bytes, *, media_type: str) -> Table:
    """A version's stored frame as a writer-ready `Table`: the frame1 document when `media_type`
    is `FRAME1_MEDIA_TYPE`, the canonical frame otherwise."""
    if media_type == FRAME1_MEDIA_TYPE:
        return read_frame1_table(data)
    return read_canonical_frame(data)


def canonical_frame_rows(table: pa.Table) -> Iterator[tuple[object, ...]]:
    """The rows of a canonical frame table, whole or projected, as `Frame.rows` gives them, a
    `unit` cell as its grammar cell. Builds `_ROWS_SLICE` rows of Python objects at once."""
    for start in range(0, table.num_rows, _ROWS_SLICE):
        row_slice = table.slice(start, _ROWS_SLICE)
        yield from zip(
            *(_column_cells(row_slice, name, json_form=False) for name in row_slice.column_names),
            strict=True,
        )


def canonical_frame_document(table: pa.Table) -> dict[str, object]:
    """The frame1 document of a canonical frame table, as `json.loads` parses its canonical
    bytes: a whole-number value as an `int`, a unit as its JSON form. Each distinct string and
    unit is one shared object, so the rows hold little beyond their lists."""
    columns = table.column_names
    cells = [_column_cells(table, name, json_form=True) for name in columns]
    return {
        "columns": list(columns),
        "frame_version": FRAME_VERSION,
        "rows": [list(row) for row in zip(*cells, strict=True)],
    }


def _column_cells(table: pa.Table, name: str, *, json_form: bool) -> list[object]:
    """One column's cells, each distinct value one shared object. A `unit` cell is its grammar
    cell, or with `json_form` its JSON form as parsed JCS. With `json_form` a value cell is
    parsed JCS too (`_jcs_number`)."""
    column = table.column(name)
    if name == UNIT_COLUMN:
        codes = unit_codes(column)
        cells = unit_cells(column, codes)
        distinct: list[object] = (
            [_parsed_jcs(unit_cell_to_json(cell)) for cell in cells] if json_form else list(cells)
        )
        return [distinct[cast(int, code)] for code in codes.codes.to_pylist()]
    if name == VALUE_COLUMN:
        values = column.to_pylist()
        return [_jcs_number(value) for value in values] if json_form else values
    if pa.types.is_dictionary(column.type) or pa.types.is_string(column.type):
        return _string_cells(column)
    return column.to_pylist()


def _string_cells(column: pa.ChunkedArray) -> list[object]:
    """A string column's cells, plain or dictionary encoded, each distinct string one object."""
    shared: dict[str, str] = {}
    cells: list[object] = []
    for chunk in column.chunks:
        encoded = cast(
            pa.DictionaryArray,
            chunk if pa.types.is_dictionary(chunk.type) else chunk.dictionary_encode(),
        )
        distinct = [
            None if value is None else shared.setdefault(value, value)
            for value in encoded.dictionary.to_pylist()
        ]
        cells.extend(
            None if index is None else distinct[index] for index in encoded.indices.to_pylist()
        )
    return cells


_JCS_EXPONENT_FROM = 1e21
"""JCS writes a number of this magnitude or more in exponent form, which parses as a float."""


def _jcs_number(value: float | None) -> float | int | None:
    """`value` as `json.loads` parses its JCS text: a whole number below `_JCS_EXPONENT_FROM` in
    magnitude as an `int`, any other number as the same float.

    JCS writes the shortest digits that round-trip to the double, as `repr` does, so the
    `int` is the one those digits spell. From 2**53 upward it can differ from `int(value)`,
    the double's exact value."""
    if value is not None and value.is_integer() and abs(value) < _JCS_EXPONENT_FROM:
        return int(Decimal(repr(value)))
    return value


def _parsed_jcs(value: object) -> object:
    """`value` as `json.loads` parses its JCS text, with JCS's sorted mapping keys."""
    if isinstance(value, Mapping):
        mapping = cast("Mapping[str, object]", value)
        return {key: _parsed_jcs(mapping[key]) for key in sorted(mapping)}
    return value
