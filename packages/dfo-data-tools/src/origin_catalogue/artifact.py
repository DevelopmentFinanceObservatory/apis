"""Writes and reads a published origin catalogue build.

The catalogue job encodes `CatalogueTables` with `encode_build` and publishes
them with `publish_build`; dfo-apis loads the published build with
`load_catalogue`. Small tables are JSON and large ones Parquet:

- `provenances.json`: one object per admitted provenance (about 160), with
  its parent Dataset as `dataset` (`{ref, label}`, or null).
- `dfo_variables.parquet`: one `variable` column (about 22,000 rows).
- `base_variables.parquet`: `variable`, `provenances` (the admitted base
  provenances holding it) and `places`, a little-endian bitset over the scan
  places, `ceil(len(scan_places) / 8)` bytes wide (up to about 160,000 rows).
- `provenance_places.parquet`: `provenance` and `places`, the scan places
  where the provenance holds a base variable, as the same bitset (one row
  per base provenance holding a base variable, about 150).
- `withheld_variables.parquet`: one `variable` column, the variables with
  data at a scan place under a provenance neither admitted nor DFO's, and
  none under an admitted base provenance.
- `topics.parquet`: `topic` and `leaves`, the variables the topic walk
  reaches from it, in walk order.
- `scan.json`: the ordered scan places and the levels scanned per source.
- `un.json`: `blocked_places`, the places the UN front door refuses to have
  named in an observation GET.
- `dataset_facts.json`: one object per DFO and base source-index entry
  (`catalogue.entry_ref`): its `ref` and its `CardFacts` members.
- `manifest.json`: `CatalogueManifest`, written last, so a build directory
  holding one is complete.

`encode_build` decodes what it encoded before returning, so a build the job
can publish is one the loader accepts, and it measures the decoded tables to
record `resident_bytes`.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from typing import Any, Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from artifact_publication.uri import join_uri, read_uri_bytes, write_uri_bytes_if_unchanged
from origin_catalogue.catalogue import (
    BaseVariable,
    CardFacts,
    CatalogueSnapshot,
    CatalogueTableError,
    CatalogueTables,
    DatasetRef,
    ProvenanceRow,
    SourceRef,
)
from origin_catalogue.origins import ORIGINS, DataCommons
from origin_catalogue.schema import (
    BASE_VARIABLES_FILE,
    CATALOGUE_FORMAT_VERSION,
    DATASET_FACTS_FILE,
    DFO_VARIABLES_FILE,
    FORMAT_ROOT,
    MANIFEST_FILE,
    POINTER_FILE,
    PROVENANCE_PLACES_FILE,
    PROVENANCES_FILE,
    SCAN_FILE,
    TOPICS_FILE,
    UN_FILE,
    VERSIONS_PREFIX,
    WITHHELD_VARIABLES_FILE,
    CatalogueManifest,
    CataloguePointer,
    ScanLevel,
    UnCarryForward,
)

_PARQUET_COMPRESSION = "zstd"

_VARIABLE_SCHEMA: Final = pa.schema([pa.field("variable", pa.string(), nullable=False)])
_BASE_VARIABLES_SCHEMA: Final = pa.schema(
    [
        pa.field("variable", pa.string(), nullable=False),
        pa.field("provenances", pa.list_(pa.string()), nullable=False),
        pa.field("places", pa.binary(), nullable=False),
    ]
)
_PROVENANCE_PLACES_SCHEMA: Final = pa.schema(
    [
        pa.field("provenance", pa.string(), nullable=False),
        pa.field("places", pa.binary(), nullable=False),
    ]
)
_TOPICS_SCHEMA: Final = pa.schema(
    [
        pa.field("topic", pa.string(), nullable=False),
        pa.field("leaves", pa.list_(pa.string()), nullable=False),
    ]
)

_FILES: Final[dict[str, str]] = {
    "provenances": PROVENANCES_FILE,
    "dfo_variables": DFO_VARIABLES_FILE,
    "base_variables": BASE_VARIABLES_FILE,
    "provenance_places": PROVENANCE_PLACES_FILE,
    "withheld_variables": WITHHELD_VARIABLES_FILE,
    "topics": TOPICS_FILE,
    "scan": SCAN_FILE,
    "un": UN_FILE,
    "dataset_facts": DATASET_FACTS_FILE,
}


class CatalogueArtifactError(Exception):
    """A published catalogue is missing, malformed, or disagrees with its manifest."""


class CatalogueTooLargeError(CatalogueArtifactError):
    """A build's `resident_bytes` exceeds the loader's memory budget."""


@dataclass(frozen=True, slots=True)
class EncodedBuild:
    """A build ready to publish: its manifest and each table file's bytes."""

    manifest: CatalogueManifest
    files: Mapping[str, bytes]


def format_root(base_uri: str) -> str:
    """This format's root under `base_uri`, the only prefix read or written."""
    return join_uri(base_uri, FORMAT_ROOT)


# --- Writing ---------------------------------------------------------------


def encode_build(
    tables: CatalogueTables,
    *,
    build_id: str,
    built_at: datetime,
    build_seconds: float,
    identity_inputs: Mapping[str, str],
    registry_hash: str,
    un_carried_forward: UnCarryForward | None = None,
) -> EncodedBuild:
    """Encodes `tables` and the manifest describing them. Row counts, the
    overlap list and `resident_bytes` are computed here. `un_carried_forward`
    is recorded as given."""
    if built_at.tzinfo is None:
        raise ValueError(f"built_at {built_at!r} has no UTC offset")
    files = {
        PROVENANCES_FILE: _encode_provenances(tables.provenances),
        DFO_VARIABLES_FILE: _encode_variables(tables.dfo_variables),
        BASE_VARIABLES_FILE: _encode_base_variables(tables),
        PROVENANCE_PLACES_FILE: _encode_provenance_places(tables),
        WITHHELD_VARIABLES_FILE: _encode_variables(tables.withheld_variables),
        TOPICS_FILE: _encode_parquet(
            pa.table(
                {
                    "topic": sorted(tables.topics),
                    "leaves": [list(tables.topics[t]) for t in sorted(tables.topics)],
                },
                schema=_TOPICS_SCHEMA,
            )
        ),
        SCAN_FILE: _encode_scan(tables),
        UN_FILE: json.dumps(
            {"blocked_places": sorted(tables.un_blocked_places)}, sort_keys=True
        ).encode("utf-8"),
        DATASET_FACTS_FILE: _encode_dataset_facts(tables.dataset_facts),
    }
    decoded = _decode_tables(files, label=f"build {build_id!r} before publication")
    manifest = CatalogueManifest(
        build_id=build_id,
        built_at=built_at,
        build_seconds=build_seconds,
        identity_inputs=dict(identity_inputs),
        registry_hash=registry_hash,
        row_counts=_row_counts(tables),
        resident_bytes=resident_bytes(decoded),
        overlaps=tuple(sorted(tables.dfo_variables & tables.base_variables.keys())),
        files=dict(_FILES),
        un_carried_forward=un_carried_forward,
    )
    return EncodedBuild(manifest=manifest, files=files)


def publish_build(base_uri: str, build: EncodedBuild, *, expected_generation: str | None) -> bool:
    """Writes the build directory, manifest last, then commits `current.json`
    if its generation still equals `expected_generation` (`None` means absent).
    Every build file is create-only, and an existing one raises
    `CatalogueArtifactError`. Returns whether the pointer write landed."""
    root = format_root(base_uri)
    prefix = join_uri(root, VERSIONS_PREFIX, build.manifest.build_id)
    payloads = [*build.files.items(), (MANIFEST_FILE, build.manifest.to_json().encode("utf-8"))]
    for name, data in payloads:
        uri = join_uri(prefix, name)
        if not write_uri_bytes_if_unchanged(uri, data, expected_generation=None):
            raise CatalogueArtifactError(
                f"{uri} already exists; a build directory is written once, by the run that "
                "created it"
            )
    pointer = CataloguePointer(build_id=build.manifest.build_id, identity=build.manifest.identity)
    return write_uri_bytes_if_unchanged(
        join_uri(root, POINTER_FILE),
        pointer.to_json().encode("utf-8"),
        expected_generation=expected_generation,
    )


def _encode_parquet(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink, compression=_PARQUET_COMPRESSION)
    return sink.getvalue()


def _encode_variables(variables: frozenset[str]) -> bytes:
    return _encode_parquet(pa.table({"variable": sorted(variables)}, schema=_VARIABLE_SCHEMA))


def _encode_provenances(provenances: Mapping[str, ProvenanceRow]) -> bytes:
    rows = [
        {
            "id": row.id,
            "data_commons": row.data_commons.id,
            "source": {"ref": row.source.ref, "label": row.source.label},
            "name": row.name,
            "url": row.url,
            "segment": row.segment,
            "dataset": (
                None
                if row.dataset is None
                else {"ref": row.dataset.ref, "label": row.dataset.label}
            ),
        }
        for row in sorted(provenances.values(), key=lambda row: row.id)
    ]
    return json.dumps(rows, sort_keys=True).encode("utf-8")


def _encode_dataset_facts(dataset_facts: Mapping[str, CardFacts]) -> bytes:
    rows = [
        {
            "ref": ref,
            "description": facts.description,
            "license_type": facts.license_type,
            "update_cadence": facts.update_cadence,
            "years_covered": facts.years_covered,
            "source_url": facts.source_url,
        }
        for ref, facts in sorted(dataset_facts.items())
    ]
    return json.dumps(rows, sort_keys=True).encode("utf-8")


def _encode_base_variables(tables: CatalogueTables) -> bytes:
    width = _bitset_width(len(tables.scan_places))
    variables = sorted(tables.base_variables)
    return _encode_parquet(
        pa.table(
            {
                "variable": variables,
                "provenances": [list(tables.base_variables[v].provenances) for v in variables],
                "places": [
                    tables.base_variables[v].places.to_bytes(width, "little") for v in variables
                ],
            },
            schema=_BASE_VARIABLES_SCHEMA,
        )
    )


def _encode_provenance_places(tables: CatalogueTables) -> bytes:
    width = _bitset_width(len(tables.scan_places))
    provenances = sorted(tables.provenance_places)
    return _encode_parquet(
        pa.table(
            {
                "provenance": provenances,
                "places": [
                    tables.provenance_places[pid].to_bytes(width, "little") for pid in provenances
                ],
            },
            schema=_PROVENANCE_PLACES_SCHEMA,
        )
    )


def _encode_scan(tables: CatalogueTables) -> bytes:
    payload = {
        "places": list(tables.scan_places),
        "levels": {
            source: [{"place_type": level.place_type, "within": level.within} for level in levels]
            for source, levels in tables.scan_levels.items()
        },
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _row_counts(tables: CatalogueTables) -> dict[str, int]:
    by_origin = {origin.id: 0 for origin in ORIGINS.values()}
    for row in tables.provenances.values():
        by_origin[row.data_commons.id] += 1
    return {
        "provenances": len(tables.provenances),
        **{f"provenances_{origin}": count for origin, count in by_origin.items()},
        "dfo_variables": len(tables.dfo_variables),
        "base_variables": len(tables.base_variables),
        "provenance_places": len(tables.provenance_places),
        "withheld_variables": len(tables.withheld_variables),
        "topics": len(tables.topics),
        "scan_places": len(tables.scan_places),
        "un_blocked_places": len(tables.un_blocked_places),
        "dataset_facts": len(tables.dataset_facts),
    }


def _bitset_width(places: int) -> int:
    return (places + 7) // 8


def resident_bytes(tables: CatalogueTables) -> int:
    """The tables' size in memory: every object reachable from them, each
    counted once, by `sys.getsizeof`."""
    seen: set[int] = set()
    stack: list[Any] = [tables]
    total = 0
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        total += sys.getsizeof(obj)
        if isinstance(obj, Mapping):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, tuple | list | set | frozenset):
            stack.extend(obj)
        elif is_dataclass(obj) and not isinstance(obj, type):
            stack.extend(getattr(obj, f.name) for f in fields(obj))
    return total


# --- Reading ---------------------------------------------------------------


def read_published_pointer(base_uri: str) -> CataloguePointer | None:
    """Reads `current.json` once. Returns `None` when it is missing or malformed."""
    uri = join_uri(format_root(base_uri), POINTER_FILE)
    try:
        return CataloguePointer.from_json(read_uri_bytes(uri))
    except (FileNotFoundError, ValueError):
        return None


def load_catalogue(
    base_uri: str, *, max_resident_bytes: int | None = None, un_system_enabled: bool = True
) -> CatalogueSnapshot:
    """Loads the build `current.json` names.

    Refuses, with `CatalogueArtifactError`, a missing or malformed pointer or
    file, a manifest of another format or build, a table whose rows disagree
    with the manifest's counts, and tables breaking `CatalogueTables`'
    invariants. Refuses with `CatalogueTooLargeError`, before reading any
    table, a build whose `resident_bytes` exceeds `max_resident_bytes`.
    """
    root = format_root(base_uri)
    uri = join_uri(root, POINTER_FILE)
    try:
        pointer = CataloguePointer.from_json(_read(uri))
    except ValueError as exc:
        raise CatalogueArtifactError(f"current.json at {uri!r} is malformed: {exc}") from exc
    prefix = join_uri(root, VERSIONS_PREFIX, pointer.build_id)
    manifest = _read_manifest(prefix)
    if manifest.build_id != pointer.build_id or manifest.identity != pointer.identity:
        raise CatalogueArtifactError(
            f"current.json at {root!r} names build {pointer.build_id!r} identity "
            f"{pointer.identity!r}, but the manifest at {prefix!r} carries build "
            f"{manifest.build_id!r} identity {manifest.identity!r}"
        )
    if max_resident_bytes is not None and manifest.resident_bytes > max_resident_bytes:
        raise CatalogueTooLargeError(
            f"catalogue build {manifest.build_id!r} needs {manifest.resident_bytes} bytes in "
            f"memory, over the {max_resident_bytes}-byte budget"
        )
    files = {
        name: _read(join_uri(prefix, _manifest_file(manifest, logical)))
        for logical, name in _FILES.items()
    }
    label = f"catalogue build {manifest.build_id!r} at {prefix!r}"
    tables = _decode_tables(files, label=label)
    counts = _row_counts(tables)
    if counts != dict(manifest.row_counts):
        raise CatalogueArtifactError(
            f"{label} has row counts {counts!r}, but its manifest records "
            f"{dict(manifest.row_counts)!r}"
        )
    return CatalogueSnapshot(manifest=manifest, tables=tables, un_system_enabled=un_system_enabled)


def read_manifest(base_uri: str, build_id: str) -> CatalogueManifest:
    """Reads one build's manifest, refusing, with `CatalogueArtifactError`,
    a missing or malformed one or one of another format."""
    return _read_manifest(join_uri(format_root(base_uri), VERSIONS_PREFIX, build_id))


def _read(uri: str) -> bytes:
    try:
        return read_uri_bytes(uri)
    except FileNotFoundError as exc:
        raise CatalogueArtifactError(f"{uri!r} is missing: {exc}") from exc


def _read_manifest(prefix: str) -> CatalogueManifest:
    uri = join_uri(prefix, MANIFEST_FILE)
    raw = _read(uri)
    try:
        format_version = json.loads(raw).get("format_version")
    except (ValueError, AttributeError) as exc:
        raise CatalogueArtifactError(f"manifest at {uri!r} is malformed: {exc}") from exc
    if format_version != CATALOGUE_FORMAT_VERSION:
        raise CatalogueArtifactError(
            f"manifest at {uri!r} has format_version {format_version!r}; this code reads format "
            f"{CATALOGUE_FORMAT_VERSION} only"
        )
    try:
        return CatalogueManifest.from_json(raw)
    except (ValueError, KeyError, TypeError) as exc:
        raise CatalogueArtifactError(f"manifest at {uri!r} is malformed: {exc!r}") from exc


def _manifest_file(manifest: CatalogueManifest, logical_name: str) -> str:
    try:
        return manifest.files[logical_name]
    except KeyError as exc:
        raise CatalogueArtifactError(
            f"manifest of build {manifest.build_id!r} names no {logical_name!r} file "
            f"(has {sorted(manifest.files)})"
        ) from exc


def _decode_tables(files: Mapping[str, bytes], *, label: str) -> CatalogueTables:
    """Decodes the table files, keyed by their standard file names."""
    try:
        provenances = _decode_provenances(files[PROVENANCES_FILE])
        scan_places, scan_levels = _decode_scan(files[SCAN_FILE])
        return CatalogueTables(
            provenances=provenances,
            dfo_variables=frozenset(
                _read_columns(files[DFO_VARIABLES_FILE], _VARIABLE_SCHEMA)["variable"]
            ),
            base_variables=_decode_base_variables(
                files[BASE_VARIABLES_FILE], provenances=provenances, places=len(scan_places)
            ),
            provenance_places=_decode_provenance_places(
                files[PROVENANCE_PLACES_FILE], places=len(scan_places)
            ),
            withheld_variables=frozenset(
                _read_columns(files[WITHHELD_VARIABLES_FILE], _VARIABLE_SCHEMA)["variable"]
            ),
            topics=_decode_topics(files[TOPICS_FILE]),
            scan_places=scan_places,
            scan_levels=scan_levels,
            dataset_facts=_decode_dataset_facts(files[DATASET_FACTS_FILE]),
            un_blocked_places=frozenset(
                _string(place) for place in json.loads(files[UN_FILE])["blocked_places"]
            ),
        )
    except (ValueError, KeyError, TypeError, pa.ArrowException) as exc:
        # CatalogueTableError is a ValueError.
        raise CatalogueArtifactError(f"{label}: {exc!r}") from exc


def _read_columns(raw: bytes, schema: pa.Schema) -> dict[str, list[Any]]:
    """Each column of `schema` as a Python list, refusing nulls at any depth."""
    table = pq.read_table(io.BytesIO(raw)).select(schema.names).cast(schema)
    columns: dict[str, list[Any]] = {}
    for name in schema.names:
        column = table.column(name)
        nulls = column.null_count
        if pa.types.is_list(column.type):
            nulls += pc.list_flatten(column).null_count
        if nulls:
            raise ValueError(f"column {name!r} holds {nulls} null(s)")
        columns[name] = column.to_pylist()
    return columns


def _decode_provenances(raw: bytes) -> dict[str, ProvenanceRow]:
    rows: dict[str, ProvenanceRow] = {}
    for item in json.loads(raw):
        row = ProvenanceRow(
            id=_string(item["id"]),
            data_commons=_origin(item["data_commons"]),
            source=SourceRef(
                ref=_string(item["source"]["ref"]), label=_string(item["source"]["label"])
            ),
            name=_string(item["name"]),
            url=_string(item["url"]),
            segment=None if item["segment"] is None else _string(item["segment"]),
            dataset=(
                None
                if (dataset := item["dataset"]) is None
                else DatasetRef(ref=_string(dataset["ref"]), label=_string(dataset["label"]))
            ),
        )
        if row.id in rows:
            raise CatalogueTableError(f"provenance {row.id!r} appears twice")
        rows[row.id] = row
    return rows


def _decode_dataset_facts(raw: bytes) -> dict[str, CardFacts]:
    facts: dict[str, CardFacts] = {}
    for item in json.loads(raw):
        ref = _string(item["ref"])
        if ref in facts:
            raise CatalogueTableError(f"dataset facts for {ref!r} appear twice")
        facts[ref] = CardFacts(
            description=_optional_string(item["description"]),
            license_type=_optional_string(item["license_type"]),
            update_cadence=_optional_string(item["update_cadence"]),
            years_covered=_optional_string(item["years_covered"]),
            source_url=_optional_string(item["source_url"]),
        )
    return facts


def _decode_base_variables(
    raw: bytes, *, provenances: Mapping[str, ProvenanceRow], places: int
) -> dict[str, BaseVariable]:
    columns = _read_columns(raw, _BASE_VARIABLES_SCHEMA)
    width = _bitset_width(places)
    # Rows share provenance ids and, mostly, whole provenance lists. Reusing one
    # object for each makes the table's memory scale with its distinct values.
    canonical_ids = {pid: pid for pid in provenances}
    canonical_lists: dict[tuple[str, ...], tuple[str, ...]] = {}
    out: dict[str, BaseVariable] = {}
    for variable, provenance_list, bitset in zip(
        columns["variable"], columns["provenances"], columns["places"], strict=True
    ):
        if len(bitset) != width:
            raise CatalogueTableError(
                f"base variable {variable!r} has a {len(bitset)}-byte place bitset; "
                f"{places} scan places need {width}"
            )
        if variable in out:
            raise CatalogueTableError(f"base variable {variable!r} appears twice")
        key = tuple(canonical_ids.get(pid, pid) for pid in provenance_list)
        out[variable] = BaseVariable(
            provenances=canonical_lists.setdefault(key, key),
            places=int.from_bytes(bitset, "little"),
        )
    return out


def _decode_provenance_places(raw: bytes, *, places: int) -> dict[str, int]:
    columns = _read_columns(raw, _PROVENANCE_PLACES_SCHEMA)
    width = _bitset_width(places)
    out: dict[str, int] = {}
    for provenance, bitset in zip(columns["provenance"], columns["places"], strict=True):
        if len(bitset) != width:
            raise CatalogueTableError(
                f"provenance {provenance!r} has a {len(bitset)}-byte place bitset; "
                f"{places} scan places need {width}"
            )
        if provenance in out:
            raise CatalogueTableError(f"provenance {provenance!r} has places twice")
        out[provenance] = int.from_bytes(bitset, "little")
    return out


def _decode_topics(raw: bytes) -> dict[str, tuple[str, ...]]:
    columns = _read_columns(raw, _TOPICS_SCHEMA)
    # About 100,000 leaves name about 30,000 variables, so each variable shares one string.
    canonical: dict[str, str] = {}
    topics: dict[str, tuple[str, ...]] = {}
    for topic, leaves in zip(columns["topic"], columns["leaves"], strict=True):
        if topic in topics:
            raise CatalogueTableError(f"topic {topic!r} appears twice")
        topics[topic] = tuple(canonical.setdefault(leaf, leaf) for leaf in leaves)
    return topics


def _decode_scan(raw: bytes) -> tuple[tuple[str, ...], dict[str, tuple[ScanLevel, ...]]]:
    payload = json.loads(raw)
    places = tuple(_string(place) for place in payload["places"])
    if len(set(places)) != len(places):
        raise CatalogueTableError("scan places repeat a place")
    levels = {
        _string(source): tuple(
            ScanLevel(
                place_type=_string(level["place_type"]),
                within=None if level["within"] is None else _string(level["within"]),
            )
            for level in entries
        )
        for source, entries in payload["levels"].items()
    }
    return places, levels


def _string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogueTableError(f"expected a non-empty string, got {value!r}")
    return value


def _optional_string(value: object) -> str | None:
    """A graph value as stored, which may be empty, or `None`."""
    if value is not None and not isinstance(value, str):
        raise CatalogueTableError(f"expected a string or null, got {value!r}")
    return value


def _origin(value: object) -> DataCommons:
    for origin in ORIGINS.values():
        if origin.id == value:
            return origin
    raise CatalogueTableError(f"data_commons {value!r} is not one of {sorted(ORIGINS)!r}")


__all__ = [
    "CatalogueArtifactError",
    "CatalogueTooLargeError",
    "EncodedBuild",
    "encode_build",
    "format_root",
    "load_catalogue",
    "publish_build",
    "read_manifest",
    "read_published_pointer",
    "resident_bytes",
]
