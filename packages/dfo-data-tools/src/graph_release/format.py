"""The graph release format: bucket layout, part schemas, manifest, writer
and reader.

A release lives under `GRAPH_RELEASE_URI`:

    parts/<sha256>.arrow | .json | .json.gz
    manifests/<release_id>.json
    format-1/current.json        {"release_id": "...", "hold": null}

A tabular part is an uncompressed Arrow IPC file holding one record batch,
sorted by the part's key, with dictionary key columns whose dictionaries are
sorted. `series` and `edges` are written one file per provenance and split
on `variable_measured` or `subject_id` boundaries at `MAX_ROWS`. A file's
name is the SHA-256 of its bytes, so a part is written once and shared by
every manifest that names it.

A reader refuses a manifest whose `format` is not `FORMAT`. It ignores part
names and columns it does not know and reads an absent optional column as
null, so a new column, part or manifest field ships without a format change.
A changed type, key, sort order or mask definition, or a removed part,
bumps `FORMAT`.

`write_part` and `write_blob_part` stage files locally and upload nothing;
`graph_release.publish` is the only code that writes to the bucket.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from artifact_publication.uri import download_uri_to_path, join_uri, read_uri_bytes

FORMAT: Final[int] = 1

MAX_ROWS = 2**21
"""The most rows a `series` or `edges` file holds, unless one variable or
subject alone has more. A module variable so a test can lower it."""

POINTER_PATH: Final[str] = f"format-{FORMAT}/current.json"
MANIFESTS_DIR: Final[str] = "manifests"
PARTS_DIR: Final[str] = "parts"

PLATFORM: Final[str] = "platform"
DFO: Final[str] = "dfo"

_KEY = pa.dictionary(pa.int32(), pa.string())
_KEY_LIST = pa.list_(_KEY)


@dataclass(frozen=True)
class PartSchema:
    """A tabular part: its columns, key (also its sort order) and the column
    a file splits on. `optional` columns may be absent from a file and read
    as null. Every other column is required."""

    name: str
    layer: str
    schema: pa.Schema
    key: tuple[str, ...]
    split_on: str | None = None
    optional: frozenset[str] = frozenset()


@dataclass(frozen=True)
class JsonPart:
    """A JSON part, decoded whole at load. Its writer serialises it. The
    suffix records whether the bytes are gzipped."""

    name: str
    layer: str
    suffix: str


SERIES = PartSchema(
    name="series",
    layer=PLATFORM,
    schema=pa.schema(
        [
            pa.field("variable_measured", _KEY, nullable=False),
            pa.field("entity1", _KEY, nullable=False),
            pa.field("extra_entities_id", _KEY, nullable=False),
            pa.field("facet_id", _KEY, nullable=False),
            pa.field("entity2", _KEY),
            pa.field("entity3", _KEY),
            pa.field("observation_count", pa.uint32(), nullable=False),
            pa.field("earliest_date", pa.date32(), nullable=False),
            pa.field("latest_date", pa.date32(), nullable=False),
            # Signed, as Spanner stores them. Readers view them as uint64 for
            # bit tests. Bit `b` of word `w` is period `64w + b` from the
            # manifest's mask epoch, null for a period the mask does not list.
            pa.field("period_mask_0", pa.int64()),
            pa.field("period_mask_1", pa.int64()),
        ]
    ),
    key=("variable_measured", "entity1", "extra_entities_id", "facet_id"),
    split_on="variable_measured",
    optional=frozenset({"entity3"}),
)

FACETS = PartSchema(
    name="facets",
    layer=PLATFORM,
    schema=pa.schema(
        [
            pa.field("facet_id", pa.string(), nullable=False),
            pa.field("provenance", _KEY, nullable=False),
            pa.field("import_name", _KEY),
            pa.field("measurement_method", _KEY),
            pa.field("observation_period", _KEY),
            pa.field("unit", _KEY),
            pa.field("scaling_factor", pa.string()),
            pa.field("is_dc_aggregate", pa.bool_(), nullable=False),
        ]
    ),
    key=("facet_id",),
)

EDGES = PartSchema(
    name="edges",
    layer=PLATFORM,
    schema=pa.schema(
        [
            pa.field("subject_id", _KEY, nullable=False),
            pa.field("predicate", _KEY, nullable=False),
            pa.field("object_id", _KEY, nullable=False),
            pa.field("provenance", _KEY, nullable=False),
        ]
    ),
    key=("subject_id", "predicate", "object_id", "provenance"),
    split_on="subject_id",
)

NODES = PartSchema(
    name="nodes",
    layer=PLATFORM,
    schema=pa.schema(
        [
            pa.field("subject_id", pa.string(), nullable=False),
            pa.field("name", pa.string()),
            pa.field("types", _KEY_LIST),
            pa.field("value", pa.string()),
            pa.field("authored", pa.bool_(), nullable=False),
        ]
    ),
    key=("subject_id",),
)

IMPORTS = PartSchema(
    name="imports",
    layer=PLATFORM,
    schema=pa.schema(
        [
            pa.field("import_name", pa.string(), nullable=False),
            # Null until imports are mapped to provenances.
            pa.field("provenance", pa.string()),
            pa.field("version", pa.string()),
            pa.field("state", pa.string()),
            pa.field("data_import_timestamp", pa.timestamp("us", tz="UTC")),
        ]
    ),
    key=("import_name",),
)

PART_SCHEMAS: Final[Mapping[str, PartSchema]] = {
    part.name: part for part in (SERIES, FACETS, EDGES, NODES, IMPORTS)
}

JSON_PARTS: Final[Mapping[str, JsonPart]] = {
    part.name: part
    for part in (
        JsonPart(name="declarations", layer=DFO, suffix=".json"),
        JsonPart(name="code_index_blocks", layer=DFO, suffix=".json.gz"),
    )
}

_ARROW_SUFFIX: Final[str] = ".arrow"

_SHA256_HEX: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_SUFFIX: Final[re.Pattern[str]] = re.compile(r"(\.[a-z0-9]+)+")
"""A part suffix: dot-led lowercase alphanumeric segments, so a part of a
kind this reader does not know may carry a new one."""


class ReleaseFormatError(Exception):
    """A release this reader must refuse: a foreign format, a malformed
    pointer or manifest, a part whose bytes do not match its hash, or a part
    missing a required column."""


MASK_EPOCH: Final[str] = "1960-01-01"
MASK_WORDS: Final[int] = 2
MASK_PERIODS: Final[tuple[str, ...]] = ("P1Y",)
"""The period mask: `MASK_WORDS` int64 columns counting periods of each
listed `observation_period` from `MASK_EPOCH`."""


@dataclass(frozen=True)
class PartRef:
    """One part file as a manifest lists it. `rows` is `None` and `columns`
    empty for a JSON part. `key_range` is the first and last value of the
    split column for `series` and `edges` files.

    Raises `ReleaseFormatError` when `sha256` is not 64 lowercase hex
    characters or `suffix` is not dot-led lowercase alphanumeric segments,
    so `file_name` is always one plain name under `parts/` and the local
    directory."""

    name: str
    layer: str
    sha256: str
    bytes: int
    rows: int | None
    columns: tuple[str, ...]
    source: str
    produced_by_execution: str
    provenance: str | None = None
    key_range: tuple[str, str] | None = None
    suffix: str = ".arrow"

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str) or not _SHA256_HEX.fullmatch(self.sha256):
            raise ReleaseFormatError(
                f"part {self.name!r} has sha256 {self.sha256!r}, not 64 lowercase hex characters"
            )
        if not isinstance(self.suffix, str) or not _SUFFIX.fullmatch(self.suffix):
            raise ReleaseFormatError(
                f"part {self.name!r} has suffix {self.suffix!r}, "
                "not dot-led lowercase alphanumeric segments"
            )

    @property
    def file_name(self) -> str:
        return f"{self.sha256}{self.suffix}"

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> PartRef:
        """Reads a manifest entry, ignoring fields this code does not know."""
        key_range = payload.get("key_range")
        return cls(
            name=payload["name"],
            layer=payload["layer"],
            sha256=payload["sha256"],
            bytes=payload["bytes"],
            rows=payload.get("rows"),
            columns=tuple(payload.get("columns") or ()),
            source=payload["source"],
            produced_by_execution=payload["produced_by_execution"],
            provenance=payload.get("provenance"),
            key_range=(key_range[0], key_range[1]) if key_range is not None else None,
            suffix=payload["suffix"],
        )


@dataclass(frozen=True)
class StagedPart:
    """A part file written to the staging directory, not yet uploaded."""

    path: Path
    ref: PartRef


def compute_release_id(
    *,
    part_hashes: Iterable[str],
    import_execution_id: str,
    dc_serving_revision: str,
    format_version: int = FORMAT,
) -> str:
    """The SHA-256 of the sorted part hashes, the import execution id, the
    serving revision and the format. Rebuilding one import against one mixer
    revision gives one id; the read timestamp and gate results stay out."""
    payload = {
        "dc_serving_revision": dc_serving_revision,
        "format": format_version,
        "import_execution_id": import_execution_id,
        "parts": sorted(part_hashes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _part_order(ref: PartRef) -> tuple[str, str, str, str]:
    return (ref.name, ref.provenance or "", ref.key_range[0] if ref.key_range else "", ref.sha256)


@dataclass(frozen=True, kw_only=True)
class Manifest:
    """One release. `release_id` derives from the parts and inputs, so a
    manifest cannot name a release its contents do not make. `parts` is kept
    in name, provenance and key order, which is the order a reader
    concatenates a part's files in. `base_un_build_id` is `None` until the
    base and UN catalogue joins the release."""

    format: int = FORMAT
    import_execution_id: str
    import_completed_at: str
    ingested_imports: tuple[str, ...]
    read_timestamp: str
    dc_serving_revision: str
    builder_code_hash: str
    declarations_hash: str
    base_un_build_id: str | None = None
    mask_epoch: str = MASK_EPOCH
    mask_words: int = MASK_WORDS
    mask_periods: tuple[str, ...] = MASK_PERIODS
    parts: tuple[PartRef, ...]
    gates: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(sorted(self.parts, key=_part_order)))
        object.__setattr__(self, "ingested_imports", tuple(self.ingested_imports))
        object.__setattr__(self, "mask_periods", tuple(self.mask_periods))

    @cached_property
    def release_id(self) -> str:
        return compute_release_id(
            part_hashes=(ref.sha256 for ref in self.parts),
            import_execution_id=self.import_execution_id,
            dc_serving_revision=self.dc_serving_revision,
            format_version=self.format,
        )

    def to_json(self) -> bytes:
        """Serialises with sorted keys, so equal manifests give equal bytes."""
        payload = dataclasses.asdict(self) | {
            "release_id": self.release_id,
            "gates": dict(self.gates),
        }
        return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> Manifest:
        """Parses a manifest, refusing a foreign `format` before reading
        anything else and a stored `release_id` its contents do not give.
        Unknown fields are ignored."""
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ReleaseFormatError(f"manifest is not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReleaseFormatError("manifest is not a JSON object")
        if payload.get("format") != FORMAT:
            raise ReleaseFormatError(
                f"manifest has format {payload.get('format')!r}; this reader reads format {FORMAT}"
            )
        try:
            manifest = cls(
                import_execution_id=payload["import_execution_id"],
                import_completed_at=payload["import_completed_at"],
                ingested_imports=tuple(payload["ingested_imports"]),
                read_timestamp=payload["read_timestamp"],
                dc_serving_revision=payload["dc_serving_revision"],
                builder_code_hash=payload["builder_code_hash"],
                declarations_hash=payload["declarations_hash"],
                base_un_build_id=payload.get("base_un_build_id"),
                mask_epoch=payload["mask_epoch"],
                mask_words=payload["mask_words"],
                mask_periods=tuple(payload["mask_periods"]),
                parts=tuple(PartRef.from_json(entry) for entry in payload["parts"]),
                gates=payload.get("gates") or {},
            )
            stored_id = payload["release_id"]
        except (KeyError, TypeError) as exc:
            raise ReleaseFormatError(f"manifest is malformed: {exc!r}") from exc
        if stored_id != manifest.release_id:
            raise ReleaseFormatError(
                f"manifest names release {stored_id!r} but its contents give "
                f"{manifest.release_id!r}"
            )
        return manifest


@dataclass(frozen=True)
class Pointer:
    """`format-<FORMAT>/current.json`. `hold`, when set, names a release the
    builder must leave serving."""

    release_id: str
    hold: str | None = None

    def to_json(self) -> bytes:
        return json.dumps(
            {"release_id": self.release_id, "hold": self.hold}, sort_keys=True
        ).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> Pointer:
        try:
            payload = json.loads(data)
            release_id, hold = payload["release_id"], payload.get("hold")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ReleaseFormatError(f"release pointer is malformed: {exc!r}") from exc
        if not isinstance(release_id, str) or not (hold is None or isinstance(hold, str)):
            raise ReleaseFormatError(f"release pointer is malformed: {payload!r}")
        return cls(release_id=release_id, hold=hold)


@dataclass(frozen=True)
class LoadedRelease:
    """A release mapped from `local_dir`. `tables` maps a part name to its
    files concatenated in manifest order, in this reader's schema. `blobs`
    maps a JSON part name to its file bytes, gzipped where its suffix says.
    `file_names` lists the files in `local_dir` the release reads, which
    leaves out the parts this reader does not know."""

    manifest: Manifest
    tables: Mapping[str, pa.Table]
    blobs: Mapping[str, bytes]
    local_dir: Path
    file_names: frozenset[str]


def pointer_uri(root_uri: str) -> str:
    return join_uri(root_uri, POINTER_PATH)


def manifest_uri(root_uri: str, release_id: str) -> str:
    return join_uri(root_uri, MANIFESTS_DIR, f"{release_id}.json")


def part_uri(root_uri: str, ref: PartRef) -> str:
    return join_uri(root_uri, PARTS_DIR, ref.file_name)


# ---------------------------------------------------------------- writing


def write_part(
    table: pa.Table,
    *,
    staging_dir: Path,
    name: str,
    layer: str,
    source: str,
    produced_by_execution: str,
    provenance: str | None = None,
) -> list[StagedPart]:
    """Encodes, sorts and splits `table` per the format rules and writes each
    file to `staging_dir` as `<sha256>.arrow`.

    `table` holds the part's columns in any order, strings either plain or
    dictionary-encoded; an optional column may be left out. `series` and
    `edges` take one provenance per call, and an `edges` table must hold
    only that provenance. Raises `ValueError` for an unknown part, a layer
    that is not the part's, an unknown or missing column, or a null in a
    non-nullable column.
    """
    part = PART_SCHEMAS.get(name)
    if part is None:
        raise ValueError(f"unknown tabular part {name!r}; known: {sorted(PART_SCHEMAS)}")
    if layer != part.layer:
        raise ValueError(f"part {name!r} is in layer {part.layer!r}, not {layer!r}")
    if part.split_on is not None and provenance is None:
        raise ValueError(f"part {name!r} is written one provenance per call")
    if part.split_on is None and provenance is not None:
        raise ValueError(f"part {name!r} has no provenance files")

    table = _sort(_conform(table, part), part)
    if name == "edges" and table.num_rows:
        provenances = _dictionary(table["provenance"]).dictionary.to_pylist()
        if provenances != [provenance]:
            raise ValueError(f"edges for provenance {provenance!r} hold {provenances}")

    ranges = _file_ranges(table, part.split_on)
    staged = []
    for lo, hi in ranges:
        chunk = table.slice(lo, hi - lo)
        if len(ranges) > 1:
            chunk = _encode_dictionaries(chunk)
        key_range = None
        if part.split_on is not None and chunk.num_rows:
            first, last = chunk[part.split_on].take([0, chunk.num_rows - 1]).to_pylist()
            key_range = (str(first), str(last))
        path, sha256, size = _stage(_ipc_bytes(chunk), _ARROW_SUFFIX, staging_dir)
        ref = PartRef(
            name=name,
            layer=layer,
            sha256=sha256,
            bytes=size,
            rows=chunk.num_rows,
            columns=tuple(chunk.column_names),
            source=source,
            produced_by_execution=produced_by_execution,
            provenance=provenance,
            key_range=key_range,
            suffix=_ARROW_SUFFIX,
        )
        staged.append(StagedPart(path=path, ref=ref))
    return staged


def write_blob_part(
    data: bytes,
    *,
    staging_dir: Path,
    name: str,
    source: str,
    produced_by_execution: str,
) -> StagedPart:
    """Writes a JSON part's bytes as `<sha256><suffix>`. The caller
    serialises, and must do so deterministically (gzipped with `mtime=0`
    where the part's suffix is `.json.gz`). Raises `ValueError` for an
    unknown part."""
    part = JSON_PARTS.get(name)
    if part is None:
        raise ValueError(f"unknown JSON part {name!r}; known: {sorted(JSON_PARTS)}")
    path, sha256, size = _stage(data, part.suffix, staging_dir)
    ref = PartRef(
        name=name,
        layer=part.layer,
        sha256=sha256,
        bytes=size,
        rows=None,
        columns=(),
        source=source,
        produced_by_execution=produced_by_execution,
        suffix=part.suffix,
    )
    return StagedPart(path=path, ref=ref)


def _conform(table: pa.Table, part: PartSchema) -> pa.Table:
    """`table` in the part's column order and types, dictionaries sorted."""
    unknown = set(table.column_names) - set(part.schema.names)
    if unknown:
        raise ValueError(f"part {part.name!r} has no columns {sorted(unknown)}")
    fields, arrays = [], []
    for schema_field in part.schema:
        if schema_field.name not in table.column_names:
            if schema_field.name in part.optional:
                continue
            raise ValueError(f"part {part.name!r} is missing column {schema_field.name!r}")
        column = table[schema_field.name].combine_chunks()
        if not schema_field.nullable and column.null_count:
            raise ValueError(f"part {part.name!r} column {schema_field.name!r} holds nulls")
        column = _encode(column, schema_field.type)
        fields.append(schema_field)
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _encode(column: pa.Array, target: pa.DataType) -> pa.Array:
    """`column` as `target`, building sorted dictionaries for dictionary
    columns and for the values of `nodes.types`."""
    if pa.types.is_dictionary(target):
        return _sorted_dictionary(column)
    if target == _KEY_LIST:
        lists = cast("pa.ListArray", column)
        return pa.ListArray.from_arrays(
            lists.offsets, _sorted_dictionary(lists.values), mask=lists.is_null()
        )
    return column.cast(target)


def _sorted_dictionary(column: pa.Array) -> pa.DictionaryArray:
    """Dictionary-encodes with a sorted dictionary and int32 indices.
    `dictionary_encode` orders values by first appearance, so the dictionary
    is built and indexed explicitly."""
    if pa.types.is_dictionary(column.type):
        column = cast("pa.DictionaryArray", column).dictionary_decode()
    column = column.cast(pa.string())
    dictionary = pc.unique(column).drop_null()
    dictionary = dictionary.take(pc.sort_indices(dictionary))
    indices = pc.index_in(column, value_set=dictionary).cast(pa.int32())
    return pa.DictionaryArray.from_arrays(indices, dictionary)


def _dictionary(column: pa.ChunkedArray) -> pa.DictionaryArray:
    return cast("pa.DictionaryArray", column.combine_chunks())


def _encode_dictionaries(table: pa.Table) -> pa.Table:
    """Re-encodes each dictionary column so its dictionary holds only the
    values `table` uses."""
    columns = [
        _sorted_dictionary(column.combine_chunks())
        if pa.types.is_dictionary(column.type)
        else column
        for column in table.columns
    ]
    return pa.Table.from_arrays(columns, schema=table.schema)


def _sort(table: pa.Table, part: PartSchema) -> pa.Table:
    # Dictionaries are sorted, so ordering by index orders by value.
    sort_columns = {}
    for key in part.key:
        column = table[key]
        sort_columns[key] = (
            _dictionary(column).indices if pa.types.is_dictionary(column.type) else column
        )
    order = pc.sort_indices(
        pa.table(sort_columns), sort_keys=[(key, "ascending") for key in part.key]
    )
    return table.take(order)


def _file_ranges(table: pa.Table, split_on: str | None) -> list[tuple[int, int]]:
    """Row ranges of at most `MAX_ROWS`, cut where the split column changes.
    A value with more rows than `MAX_ROWS` keeps them in one file."""
    rows = table.num_rows
    if split_on is None or rows <= MAX_ROWS:
        return [(0, rows)]
    keys = _dictionary(table[split_on]).indices.to_numpy()
    ranges = []
    lo = 0
    while lo < rows:
        hi = lo + MAX_ROWS
        if hi < rows:
            cut = int(np.searchsorted(keys, keys[hi], side="left"))
            hi = cut if cut > lo else int(np.searchsorted(keys, keys[hi], side="right"))
        hi = min(hi, rows)
        ranges.append((lo, hi))
        lo = hi
    return ranges


def _ipc_bytes(table: pa.Table) -> pa.Buffer:
    batch = pa.RecordBatch.from_arrays(
        [column.combine_chunks() for column in table.columns], schema=table.schema
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _stage(data: bytes | pa.Buffer, suffix: str, staging_dir: Path) -> tuple[Path, str, int]:
    sha256 = hashlib.sha256(data).hexdigest()
    path = staging_dir / f"{sha256}{suffix}"
    if not path.exists():
        staging_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path, sha256, len(data)


# ---------------------------------------------------------------- reading


def read_pointer(root_uri: str) -> Pointer | None:
    """The format's pointer, or `None` when no release is published."""
    try:
        data = read_uri_bytes(pointer_uri(root_uri))
    except FileNotFoundError:
        return None
    return Pointer.from_json(data)


def open_release(root_uri: str, *, local_dir: Path, release_id: str | None = None) -> LoadedRelease:
    """Opens the pointer's release, or `release_id` when given.

    Downloads each part this reader knows and `local_dir` lacks with one
    streamed GET, checks its SHA-256, and maps it. Raises `FileNotFoundError`
    when no pointer is published and `ReleaseFormatError` for a release this
    reader must refuse.
    """
    if release_id is None:
        pointer = read_pointer(root_uri)
        if pointer is None:
            raise FileNotFoundError(f"no release pointer at {pointer_uri(root_uri)}")
        release_id = pointer.release_id
    manifest = Manifest.from_json(read_uri_bytes(manifest_uri(root_uri, release_id)))
    if manifest.release_id != release_id:
        raise ReleaseFormatError(f"manifest {release_id!r} holds release {manifest.release_id!r}")

    local_dir.mkdir(parents=True, exist_ok=True)
    pieces: dict[str, list[pa.Table]] = {}
    blobs: dict[str, bytes] = {}
    file_names: set[str] = set()
    for ref in manifest.parts:
        if ref.name in PART_SCHEMAS:
            path = _fetch(root_uri, ref, local_dir)
            pieces.setdefault(ref.name, []).append(_map(path, PART_SCHEMAS[ref.name]))
        elif ref.name in JSON_PARTS:
            if ref.name in blobs:
                raise ReleaseFormatError(f"manifest lists JSON part {ref.name!r} twice")
            blobs[ref.name] = _fetch(root_uri, ref, local_dir).read_bytes()
        else:
            continue
        file_names.add(ref.file_name)
    tables = {name: pa.concat_tables(files) for name, files in pieces.items()}
    return LoadedRelease(
        manifest=manifest,
        tables=tables,
        blobs=blobs,
        local_dir=local_dir,
        file_names=frozenset(file_names),
    )


def _fetch(root_uri: str, ref: PartRef, local_dir: Path) -> Path:
    """The part's local path, downloading it first when absent. A file only
    takes its final name after its hash checks, so a present file is
    trusted."""
    path = local_dir / ref.file_name
    if path.exists():
        return path
    with tempfile.NamedTemporaryFile(dir=local_dir, suffix=".partial", delete=False) as handle:
        partial = Path(handle.name)
    try:
        download_uri_to_path(part_uri(root_uri, ref), partial)
        with partial.open("rb") as downloaded:
            digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
        if digest != ref.sha256:
            raise ReleaseFormatError(
                f"part {ref.name!r} downloaded with SHA-256 {digest}, manifest says {ref.sha256}"
            )
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _map(path: Path, part: PartSchema) -> pa.Table:
    """Maps one file into the part's schema: unknown columns dropped, absent
    optional columns null."""
    table = pa.ipc.open_file(pa.memory_map(str(path), "r")).read_all()
    columns = []
    for schema_field in part.schema:
        if schema_field.name in table.column_names:
            column = table[schema_field.name]
            if column.type != schema_field.type:
                raise ReleaseFormatError(
                    f"part {part.name!r} column {schema_field.name!r} is {column.type}, "
                    f"this reader reads {schema_field.type}"
                )
            columns.append(column)
        elif schema_field.name in part.optional:
            columns.append(pa.nulls(table.num_rows, schema_field.type))
        else:
            raise ReleaseFormatError(
                f"part {part.name!r} file {path.name} lacks column {schema_field.name!r}"
            )
    return pa.Table.from_arrays(columns, schema=part.schema)


__all__ = [
    "DFO",
    "EDGES",
    "FACETS",
    "FORMAT",
    "IMPORTS",
    "JSON_PARTS",
    "MANIFESTS_DIR",
    "MASK_EPOCH",
    "MASK_PERIODS",
    "MASK_WORDS",
    "MAX_ROWS",
    "NODES",
    "PARTS_DIR",
    "PART_SCHEMAS",
    "PLATFORM",
    "POINTER_PATH",
    "SERIES",
    "JsonPart",
    "LoadedRelease",
    "Manifest",
    "PartRef",
    "PartSchema",
    "Pointer",
    "ReleaseFormatError",
    "StagedPart",
    "compute_release_id",
    "manifest_uri",
    "open_release",
    "part_uri",
    "pointer_uri",
    "read_pointer",
    "write_blob_part",
    "write_part",
]
