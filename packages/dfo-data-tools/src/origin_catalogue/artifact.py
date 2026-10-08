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
- `superseded_variables.parquet`: `variable` and `successor`, each withheld
  variable of a base provenance a UN agency supersedes and the admitted UN
  variable that stands for it, null where none does
  (`origin_catalogue.successors`).
- `manifest.json`: `CatalogueManifest`, written last, so a build directory
  holding one is complete.

The UN tables (`un_mirror.UnMirror`) are five more files, which
`load_catalogue` reads into the snapshot beside the other tables:

- `un_variables.parquet`: one row per admitted variable of a registered UN
  agency (about 84,000): `variable`, its `series` (the `populationType`
  target) and `arcs`, row numbers into `un_arcs.parquet`. Together the two
  answer `node_arcs_batch` for a UN variable, and with it every fact shape
  derivation reads from the arcs: the five-tuple, the constraint properties
  and values, `observationProperties`, the name and `typeOf`. The rows of
  one `series` are the members a UN series shape takes its slot keys from.
- `un_arcs.parquet`: one row per distinct arc target (`un_mirror.UnArc`):
  `property`, the target's `dcid` or literal `value`, its `name` and its
  `types`. A row's `name` answers `node_labels_batch` for a constraint
  property, a constraint value and a series.
- `un_excluded_variables.parquet`: `variable` and `series`, every member
  variable of a series the registry excludes. Admission refuses these whole
  dcids. They have no row in `un_variables.parquet`.
- `un_topics.parquet`: `topic` and `leaves`, each UN topic under
  `undata/topic/Root` and the admitted variables the topic walk reaches from
  it, in walk order. It answers the topic expansion of a UN topic.
- `un_mirror.json`: `exclusions`, each excluded series and the DFO
  provenance its registry entry names, `series_names`, each excluded
  series and its name, and `agencies`, each agency's
  `UnAgencyCounts` by id segment.

`encode_build` decodes what it encoded before returning, so a build the job
can publish is one the loader accepts, and it measures the decoded tables to
record `resident_bytes` and the UN tables' `un_mirror.resident_bytes`.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Iterator, Mapping
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
    UnTables,
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
    SUPERSEDED_VARIABLES_FILE,
    TOPICS_FILE,
    UN_ARCS_FILE,
    UN_EXCLUDED_VARIABLES_FILE,
    UN_FILE,
    UN_MIRROR_FILE,
    UN_TOPICS_FILE,
    UN_VARIABLES_FILE,
    VERSIONS_PREFIX,
    WITHHELD_VARIABLES_FILE,
    CatalogueManifest,
    CataloguePointer,
    ScanLevel,
    UnCarryForward,
    UnMirrorManifest,
    agency_counts_from_json,
)
from origin_catalogue.successors import unheld_successors
from origin_catalogue.un_mirror import (
    EMPTY_UN_MIRROR,
    UnArc,
    UnMirror,
    UnVariable,
    foreign_variables,
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
_UN_VARIABLES_SCHEMA: Final = pa.schema(
    [
        pa.field("variable", pa.string(), nullable=False),
        pa.field("series", pa.string(), nullable=False),
        pa.field("arcs", pa.list_(pa.int32()), nullable=False),
    ]
)
_UN_ARCS_SCHEMA: Final = pa.schema(
    [
        pa.field("property", pa.string(), nullable=False),
        pa.field("dcid", pa.string()),
        pa.field("value", pa.string()),
        pa.field("name", pa.string()),
        pa.field("types", pa.list_(pa.string()), nullable=False),
    ]
)
_UN_EXCLUDED_VARIABLES_SCHEMA: Final = pa.schema(
    [
        pa.field("variable", pa.string(), nullable=False),
        pa.field("series", pa.string(), nullable=False),
    ]
)
_SUPERSEDED_VARIABLES_SCHEMA: Final = pa.schema(
    [
        pa.field("variable", pa.string(), nullable=False),
        pa.field("successor", pa.string()),
    ]
)

_UN_DECODE_ROWS: Final[int] = 4_096
"""Rows of a UN table file converted to Python objects at a time. A whole
column converted at once holds a Python object per arc reference and per
topic leaf, several times the size of the tables they decode to."""

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
    "superseded_variables": SUPERSEDED_VARIABLES_FILE,
}

_UN_MIRROR_FILES: Final[dict[str, str]] = {
    "un_variables": UN_VARIABLES_FILE,
    "un_arcs": UN_ARCS_FILE,
    "un_excluded_variables": UN_EXCLUDED_VARIABLES_FILE,
    "un_topics": UN_TOPICS_FILE,
    "un_mirror": UN_MIRROR_FILE,
}


class CatalogueArtifactError(Exception):
    """A published catalogue is missing, malformed, or disagrees with its manifest."""


class CatalogueTooLargeError(CatalogueArtifactError):
    """A build's tables, the UN tables included, exceed the loader's memory budget."""


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
    un_mirror: UnMirror = EMPTY_UN_MIRROR,
) -> EncodedBuild:
    """Encodes `tables`, `un_mirror` and the manifest describing them. Row
    counts, the overlap list and both `resident_bytes` are computed here.
    `un_carried_forward` is recorded as given. Each table set is decoded,
    compared or measured, and released before the other is decoded, so one
    decoded copy is alive at a time beside the tables being encoded."""
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
        SUPERSEDED_VARIABLES_FILE: _encode_parquet(
            pa.table(
                {
                    "variable": sorted(tables.superseded),
                    "successor": [tables.superseded[v] for v in sorted(tables.superseded)],
                },
                schema=_SUPERSEDED_VARIABLES_SCHEMA,
            )
        ),
        **_encode_un_mirror(un_mirror),
    }
    label = f"build {build_id!r} before publication"
    tables_bytes = resident_bytes(_decode_tables(files, label=label))
    un_bytes = _checked_un_bytes(files, un_mirror, label=label)
    manifest = CatalogueManifest(
        build_id=build_id,
        built_at=built_at,
        build_seconds=build_seconds,
        identity_inputs=dict(identity_inputs),
        registry_hash=registry_hash,
        row_counts=_row_counts(tables),
        resident_bytes=tables_bytes,
        overlaps=tuple(sorted(tables.dfo_variables & tables.base_variables.keys())),
        files={**_FILES, **_UN_MIRROR_FILES},
        un_carried_forward=un_carried_forward,
        un_mirror=UnMirrorManifest(
            row_counts=_un_row_counts(un_mirror),
            agencies=dict(un_mirror.agencies),
            resident_bytes=un_bytes,
        ),
    )
    return EncodedBuild(manifest=manifest, files=files)


def _checked_un_bytes(files: Mapping[str, bytes], un_mirror: UnMirror, *, label: str) -> int:
    """The size in memory of the UN tables `files` decode to, refusing files
    that do not decode to `un_mirror`."""
    decoded = decode_un_mirror(files, label=label)
    if decoded != un_mirror:
        raise CatalogueArtifactError(f"{label}: the UN tables do not decode to what was encoded")
    return resident_bytes(decoded)


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


def _encode_un_mirror(mirror: UnMirror) -> dict[str, bytes]:
    """The UN tables' five files. Arc rows are numbered in first-use order
    over the variables sorted by dcid, so equal tables encode to equal bytes."""
    variables = sorted(mirror.variables)
    rows: dict[UnArc, int] = {}
    arc_lists = [
        [rows.setdefault(arc, len(rows)) for arc in mirror.variables[dcid].arcs]
        for dcid in variables
    ]
    excluded = sorted(mirror.excluded_variables)
    topics = sorted(mirror.topics)
    return {
        UN_VARIABLES_FILE: _encode_parquet(
            pa.table(
                {
                    "variable": variables,
                    "series": [mirror.variables[dcid].series for dcid in variables],
                    "arcs": arc_lists,
                },
                schema=_UN_VARIABLES_SCHEMA,
            )
        ),
        UN_ARCS_FILE: _encode_parquet(
            pa.table(
                {
                    "property": [arc.property for arc in rows],
                    "dcid": [arc.dcid for arc in rows],
                    "value": [arc.value for arc in rows],
                    "name": [arc.name for arc in rows],
                    "types": [list(arc.types) for arc in rows],
                },
                schema=_UN_ARCS_SCHEMA,
            )
        ),
        UN_EXCLUDED_VARIABLES_FILE: _encode_parquet(
            pa.table(
                {
                    "variable": excluded,
                    "series": [mirror.excluded_variables[dcid] for dcid in excluded],
                },
                schema=_UN_EXCLUDED_VARIABLES_SCHEMA,
            )
        ),
        UN_TOPICS_FILE: _encode_parquet(
            pa.table(
                {"topic": topics, "leaves": [list(mirror.topics[topic]) for topic in topics]},
                schema=_TOPICS_SCHEMA,
            )
        ),
        UN_MIRROR_FILE: json.dumps(
            {
                "exclusions": dict(mirror.exclusions),
                "series_names": dict(mirror.excluded_series_names),
                "agencies": {
                    segment: {
                        "series": counts.series,
                        "variables": counts.variables,
                        "excluded_series": counts.excluded_series,
                        "excluded_variables": counts.excluded_variables,
                    }
                    for segment, counts in mirror.agencies.items()
                },
            },
            sort_keys=True,
        ).encode("utf-8"),
    }


def _un_row_counts(mirror: UnMirror) -> dict[str, int]:
    return {
        "un_variables": len(mirror.variables),
        "un_arcs": len({arc for variable in mirror.variables.values() for arc in variable.arcs}),
        "un_excluded_variables": len(mirror.excluded_variables),
        "un_excluded_series": len(mirror.exclusions),
        "un_series": len({variable.series for variable in mirror.variables.values()}),
        "un_topics": len(mirror.topics),
    }


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
        "superseded_variables": len(tables.superseded),
        "successor_variables": sum(
            successor is not None for successor in tables.superseded.values()
        ),
    }


def _bitset_width(places: int) -> int:
    return (places + 7) // 8


def resident_bytes(tables: CatalogueTables | UnMirror) -> int:
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
    with the manifest's counts, UN tables whose row counts or agency counts
    disagree with the manifest's, tables breaking `CatalogueTables`' or
    `UnMirror`'s invariants, and a successor the UN tables do not admit.
    Refuses with `CatalogueTooLargeError`, before
    reading any table, a build whose `resident_bytes` and
    `un_mirror.resident_bytes` together exceed `max_resident_bytes`.
    """
    manifest, tables = load_tables(base_uri, max_resident_bytes=max_resident_bytes)
    prefix = join_uri(format_root(base_uri), VERSIONS_PREFIX, manifest.build_id)
    mirror = _read_un_mirror(prefix, manifest)
    unheld = unheld_successors(tables.superseded, mirror)
    if unheld:
        raise CatalogueArtifactError(
            f"catalogue build {manifest.build_id!r} at {prefix!r}: {len(unheld)} successor(s) "
            f"are not admitted UN variables of the build, e.g. for {unheld[:3]!r}"
        )
    return CatalogueSnapshot(
        manifest=manifest,
        tables=tables,
        un=UnTables(mirror=mirror),
        un_system_enabled=un_system_enabled,
    )


def load_tables(
    base_uri: str, *, max_resident_bytes: int | None = None
) -> tuple[CatalogueManifest, CatalogueTables]:
    """The manifest and tables of the build `current.json` names, without
    its UN tables, which `read_un_mirror` reads. Refuses what
    `load_catalogue` refuses of the pointer, the manifest, the budget and
    these tables."""
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
    needed = manifest.resident_bytes + manifest.un_mirror.resident_bytes
    if max_resident_bytes is not None and needed > max_resident_bytes:
        raise CatalogueTooLargeError(
            f"catalogue build {manifest.build_id!r} needs {needed} bytes in memory "
            f"({manifest.un_mirror.resident_bytes} of them for the UN tables), over the "
            f"{max_resident_bytes}-byte budget"
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
    return manifest, tables


def read_manifest(base_uri: str, build_id: str) -> CatalogueManifest:
    """Reads one build's manifest, refusing, with `CatalogueArtifactError`,
    a missing or malformed one or one of another format."""
    return _read_manifest(join_uri(format_root(base_uri), VERSIONS_PREFIX, build_id))


def read_un_mirror(base_uri: str, manifest: CatalogueManifest) -> UnMirror:
    """Reads the UN tables of `manifest`'s build. Refuses, with
    `CatalogueArtifactError`, a missing or malformed file, tables breaking
    `UnMirror`'s invariants, and tables whose row counts or agency counts
    disagree with the manifest's."""
    return _read_un_mirror(
        join_uri(format_root(base_uri), VERSIONS_PREFIX, manifest.build_id), manifest
    )


def _read_un_mirror(prefix: str, manifest: CatalogueManifest) -> UnMirror:
    files = {
        name: _read(join_uri(prefix, _manifest_file(manifest, logical)))
        for logical, name in _UN_MIRROR_FILES.items()
    }
    label = f"catalogue build {manifest.build_id!r} at {prefix!r}"
    mirror = decode_un_mirror(files, label=label)
    if foreign := foreign_variables(mirror):
        raise CatalogueArtifactError(
            f"{label} holds {len(foreign)} UN variable(s) no agency's id prefix starts, "
            f"e.g. {foreign[:3]!r}"
        )
    recorded = manifest.un_mirror
    if _un_row_counts(mirror) != dict(recorded.row_counts) or mirror.agencies != recorded.agencies:
        raise CatalogueArtifactError(
            f"{label} has UN row counts {_un_row_counts(mirror)!r} and agencies "
            f"{dict(mirror.agencies)!r}, but its manifest records {recorded!r}"
        )
    return mirror


def decode_un_mirror(files: Mapping[str, bytes], *, label: str) -> UnMirror:
    """Decodes the UN table files, keyed by their standard file names. Rows
    share one `UnArc` per arc row and one string per property, type, series
    and variable, so the tables' memory scales with their distinct values.
    Each file is decoded `_UN_DECODE_ROWS` rows at a time, so the lists that
    exist only while decoding hold one batch."""
    try:
        properties: dict[str, str] = {}
        types: dict[str, str] = {}
        arcs = [
            UnArc(
                property=properties.setdefault(property, property),
                dcid=dcid,
                value=value,
                name=name,
                types=tuple(types.setdefault(kind, kind) for kind in kinds),
            )
            for property, dcid, value, name, kinds in _rows(files[UN_ARCS_FILE], _UN_ARCS_SCHEMA)
        ]
        series_ids: dict[str, str] = {}
        variables: dict[str, UnVariable] = {}
        for variable, series, rows in _rows(files[UN_VARIABLES_FILE], _UN_VARIABLES_SCHEMA):
            if variable in variables:
                raise CatalogueTableError(f"UN variable {variable!r} appears twice")
            if any(not 0 <= row < len(arcs) for row in rows):
                raise CatalogueTableError(f"UN variable {variable!r} names an arc row out of range")
            variables[variable] = UnVariable(
                series=series_ids.setdefault(series, series), arcs=tuple(arcs[row] for row in rows)
            )
        excluded_series: dict[str, str] = {}
        excluded: dict[str, str] = {}
        for variable, series in _rows(
            files[UN_EXCLUDED_VARIABLES_FILE], _UN_EXCLUDED_VARIABLES_SCHEMA
        ):
            if variable in excluded:
                raise CatalogueTableError(f"excluded UN variable {variable!r} appears twice")
            excluded[variable] = excluded_series.setdefault(series, series)
        payload = json.loads(files[UN_MIRROR_FILE])
        return UnMirror(
            variables=variables,
            excluded_variables=excluded,
            topics=_decode_un_topics(files[UN_TOPICS_FILE], variables),
            exclusions={
                _string(series): _string(provenance)
                for series, provenance in payload["exclusions"].items()
            },
            excluded_series_names={
                _string(series): _string(name) for series, name in payload["series_names"].items()
            },
            agencies={
                _string(segment): agency_counts_from_json(counts)
                for segment, counts in payload["agencies"].items()
            },
        )
    except (ValueError, KeyError, TypeError, AttributeError, pa.ArrowException) as exc:
        # CatalogueTableError and UnMirrorTableError are ValueErrors.
        raise CatalogueArtifactError(f"{label}: {exc!r}") from exc


def _decode_un_topics(
    raw: bytes, variables: Mapping[str, UnVariable]
) -> dict[str, tuple[str, ...]]:
    """The UN topic table, each leaf sharing its string with `variables`' key."""
    ids = {variable: variable for variable in variables}
    topics: dict[str, tuple[str, ...]] = {}
    for topic, leaves in _rows(raw, _TOPICS_SCHEMA):
        if topic in topics:
            raise CatalogueTableError(f"UN topic {topic!r} appears twice")
        topics[topic] = tuple(ids.setdefault(leaf, leaf) for leaf in leaves)
    return topics


def _rows(raw: bytes, schema: pa.Schema) -> Iterator[tuple[Any, ...]]:
    """The rows of a Parquet file cast to `schema`, one tuple of the
    schema's fields per row, read and converted `_UN_DECODE_ROWS` rows at a
    time. Refuses a null in a field the schema does not mark nullable and in
    any list's items."""
    batches = pq.ParquetFile(io.BytesIO(raw)).iter_batches(
        batch_size=_UN_DECODE_ROWS, columns=schema.names, use_threads=False
    )
    for read in batches:
        batch = read.select(schema.names).cast(schema)
        for column_field, column in zip(schema, batch.columns, strict=True):
            nulls = 0 if column_field.nullable else column.null_count
            if pa.types.is_list(column.type):
                nulls += pc.list_flatten(column).null_count
            if nulls:
                raise ValueError(f"column {column_field.name!r} holds {nulls} null(s)")
        yield from zip(*(column.to_pylist() for column in batch.columns), strict=True)


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
            superseded=_decode_superseded(files[SUPERSEDED_VARIABLES_FILE]),
        )
    except (ValueError, KeyError, TypeError, pa.ArrowException) as exc:
        # CatalogueTableError is a ValueError.
        raise CatalogueArtifactError(f"{label}: {exc!r}") from exc


def _decode_superseded(raw: bytes) -> dict[str, str | None]:
    """The superseded variables, each with its successor or `None`."""
    schema = _SUPERSEDED_VARIABLES_SCHEMA
    table = pq.read_table(io.BytesIO(raw)).select(schema.names).cast(schema)
    variables = table.column("variable")
    if variables.null_count:
        raise ValueError(f"column 'variable' holds {variables.null_count} null(s)")
    successors = table.column("successor").to_pylist()
    superseded: dict[str, str | None] = {
        _string(variable): None if successor is None else _string(successor)
        for variable, successor in zip(variables.to_pylist(), successors, strict=True)
    }
    if len(superseded) != table.num_rows:
        raise CatalogueTableError("a superseded variable appears twice")
    return superseded


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
    "decode_un_mirror",
    "encode_build",
    "format_root",
    "load_catalogue",
    "load_tables",
    "publish_build",
    "read_manifest",
    "read_published_pointer",
    "read_un_mirror",
    "resident_bytes",
]
