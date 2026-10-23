"""Writes and reads a published origin catalogue build.

The catalogue job encodes `CatalogueTables` with `encode_build` and publishes
them with `publish_build`; dfo-apis loads the published build with
`load_catalogue`. Small tables are JSON and large ones Parquet:

- `provenances.json`: one object per admitted provenance (about 160), with
  its parent Dataset as `dataset` (`{ref, label}`, or null) and its
  `licence` (`{type, label, url}`, each member a string or null).
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
- `source_documentation.json`: one object per entry of `dataset_facts.json`:
  its `ref` and its `SourceDocumentation` members, `documents` an object from
  document property to its list of stored values.
- `superseded_variables.parquet`: `variable` and `successor`, each withheld
  variable of a base provenance a UN agency supersedes and the admitted UN
  variable that stands for it, null where none does
  (`origin_catalogue.successors`).
- `manifest.json`: `CatalogueManifest`, written last, so a build directory
  holding one is complete. It records the SHA-256 of every other file, and
  the loader refuses a file whose bytes do not match, so a file from another
  build or edited by hand never loads beside the others.

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

The arc tables (`arc_tables.ArcTables`) are four more files, in the UN
tables' shape:

- `dfo_arcs.parquet`, `base_arcs.parquet` and `schema_arcs.parquet`: one row
  per node, `node` and `arcs`, row numbers into `arc_targets.parquet`. An
  empty `arcs` holds the node empty. The first two hold listed DFO and base
  variables, the third the schema nodes their arcs depend on.
- `arc_targets.parquet`: one row per distinct arc target, with the columns
  of `un_arcs.parquet` and `label`, the label of the node the row names
  (`ArcTables.labels`), null for a literal and for a node with no name.

`families.parquet` holds the family tables (`families.FamilyTables`), which
`encode_build` computes from the arc tables and the UN tables: one row per
five-tuple family and role class (`families.FamilyKey`, one column per field)
of the `dfo` and `base` variables, `un` false, and one per family of the
admitted UN variables, `un` true. Each row carries `members`, the member
dcids in table order, `properties` and `values`, each constraint property the members assert
and the distinct values they assert for it, `shared`, the properties every
member asserts at one value, and `total` and `total_name`, the family total
and its name, null where the family has none.

`encode_build` decodes what it encoded before returning, so a build the job
can publish is one the loader accepts, and it measures the decoded tables to
record `resident_bytes`, the UN tables' `un_mirror.resident_bytes`, the
arc tables' `arc_tables.resident_bytes` and the family tables'
`families.resident_bytes`.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from typing import Any, Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from artifact_publication.jobs import Ingestion
from artifact_publication.uri import join_uri, read_uri_bytes, write_uri_bytes_if_unchanged
from origin_catalogue.arc_tables import EMPTY_ARC_TABLES, ArcTables
from origin_catalogue.catalogue import (
    BaseVariable,
    CardFacts,
    CatalogueSnapshot,
    CatalogueTableError,
    CatalogueTables,
    DatasetRef,
    Licence,
    ProvenanceRow,
    SourceDocumentation,
    SourceRef,
    UnTables,
)
from origin_catalogue.families import (
    FamilyKey,
    FamilySummary,
    FamilyTables,
    family_tables,
    summarise_table,
)
from origin_catalogue.origins import ORIGINS, UN_SYSTEM, DataCommons
from origin_catalogue.schema import (
    ARC_TARGETS_FILE,
    BASE_ARCS_FILE,
    BASE_VARIABLES_FILE,
    CATALOGUE_FORMAT_VERSION,
    DATASET_FACTS_FILE,
    DFO_ARCS_FILE,
    DFO_VARIABLES_FILE,
    FAMILIES_FILE,
    FORMAT_ROOT,
    MANIFEST_FILE,
    NO_INDEX_PRESENCE,
    POINTER_FILE,
    PROVENANCE_PLACES_FILE,
    PROVENANCES_FILE,
    SCAN_FILE,
    SCHEMA_ARCS_FILE,
    SOURCE_DOCUMENTATION_FILE,
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
    ArcTablesManifest,
    CatalogueManifest,
    CataloguePointer,
    FamilyTablesManifest,
    IndexPresence,
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
_ARC_ROWS_SCHEMA: Final = pa.schema(
    [
        pa.field("node", pa.string(), nullable=False),
        pa.field("arcs", pa.list_(pa.int32()), nullable=False),
    ]
)
_ARC_TARGETS_SCHEMA: Final = pa.schema(
    [
        pa.field("property", pa.string(), nullable=False),
        pa.field("dcid", pa.string()),
        pa.field("value", pa.string()),
        pa.field("name", pa.string()),
        pa.field("types", pa.list_(pa.string()), nullable=False),
    ]
)
_ARC_TABLE_TARGETS_SCHEMA: Final = _ARC_TARGETS_SCHEMA.append(pa.field("label", pa.string()))
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

_FAMILIES_SCHEMA: Final = pa.schema(
    [
        pa.field("un", pa.bool_(), nullable=False),
        pa.field("population_type", pa.string(), nullable=False),
        pa.field("measured_property", pa.string(), nullable=False),
        pa.field("stat_type", pa.string(), nullable=False),
        pa.field("measurement_qualifier", pa.string()),
        pa.field("measurement_denominator", pa.string()),
        pa.field("denominator_unresolved", pa.bool_(), nullable=False),
        pa.field("role_bearing", pa.bool_(), nullable=False),
        pa.field("members", pa.list_(pa.string()), nullable=False),
        pa.field("properties", pa.list_(pa.string()), nullable=False),
        pa.field("values", pa.list_(pa.list_(pa.string())), nullable=False),
        pa.field("shared", pa.list_(pa.string()), nullable=False),
        pa.field("total", pa.string()),
        pa.field("total_name", pa.string()),
    ]
)

_UN_DECODE_ROWS: Final[int] = 4_096
"""Rows of a UN or arc table file converted to Python objects at a time. A whole
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
    "source_documentation": SOURCE_DOCUMENTATION_FILE,
    "superseded_variables": SUPERSEDED_VARIABLES_FILE,
}

_UN_MIRROR_FILES: Final[dict[str, str]] = {
    "un_variables": UN_VARIABLES_FILE,
    "un_arcs": UN_ARCS_FILE,
    "un_excluded_variables": UN_EXCLUDED_VARIABLES_FILE,
    "un_topics": UN_TOPICS_FILE,
    "un_mirror": UN_MIRROR_FILE,
}

_ARC_TABLES_FILES: Final[dict[str, str]] = {
    "dfo_arcs": DFO_ARCS_FILE,
    "base_arcs": BASE_ARCS_FILE,
    "schema_arcs": SCHEMA_ARCS_FILE,
    "arc_targets": ARC_TARGETS_FILE,
}

_FAMILY_FILES: Final[dict[str, str]] = {"families": FAMILIES_FILE}


class CatalogueArtifactError(Exception):
    """A published catalogue is missing, malformed, or disagrees with its manifest."""


class CatalogueTooLargeError(CatalogueArtifactError):
    """A build's tables, the UN and arc tables included, exceed the loader's memory budget."""


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
    arcs: ArcTables = EMPTY_ARC_TABLES,
    index_presence: IndexPresence = NO_INDEX_PRESENCE,
    ingestion: Ingestion | None = None,
) -> EncodedBuild:
    """Encodes `tables`, `un_mirror`, `arcs`, the family tables of `arcs` and
    `un_mirror` (`families.family_tables`) and the manifest describing them.
    Row counts, the overlap list and the four `resident_bytes` are computed
    here. `un_carried_forward`, `index_presence` and `ingestion` are
    recorded as given. Each table set
    is decoded, compared or measured, and released before the next is
    decoded, so one decoded copy is alive at a time beside the tables being
    encoded. The manifest records each file's SHA-256. Refuses, with
    `CatalogueArtifactError`, arc tables that disagree with the variables
    `tables` lists (`_check_arc_scope`)."""
    if built_at.tzinfo is None:
        raise ValueError(f"built_at {built_at!r} has no UTC offset")
    label = f"build {build_id!r} before publication"
    _check_arc_scope(tables, arcs, label=label)
    families = family_tables(dfo=arcs.dfo, base=arcs.base, un=un_mirror)
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
        SOURCE_DOCUMENTATION_FILE: _encode_source_documentation(tables.source_documentation),
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
        **_encode_arc_tables(arcs),
        FAMILIES_FILE: _encode_families(families),
    }
    tables_bytes = resident_bytes(_decode_tables(files, label=label))
    un_bytes = _checked_un_bytes(files, un_mirror, label=label)
    arc_bytes = _checked_arc_bytes(files, arcs, label=label)
    family_bytes = _checked_family_bytes(files, families, label=label)
    manifest_files = {**_FILES, **_UN_MIRROR_FILES, **_ARC_TABLES_FILES, **_FAMILY_FILES}
    manifest = CatalogueManifest(
        build_id=build_id,
        built_at=built_at,
        build_seconds=build_seconds,
        identity_inputs=dict(identity_inputs),
        registry_hash=registry_hash,
        row_counts=_row_counts(tables),
        resident_bytes=tables_bytes,
        overlaps=tuple(sorted(tables.dfo_variables & tables.base_variables.keys())),
        files=manifest_files,
        file_sha256={
            logical: hashlib.sha256(files[name]).hexdigest()
            for logical, name in manifest_files.items()
        },
        un_carried_forward=un_carried_forward,
        un_mirror=UnMirrorManifest(
            row_counts=_un_row_counts(un_mirror),
            agencies=dict(un_mirror.agencies),
            resident_bytes=un_bytes,
        ),
        arc_tables=ArcTablesManifest(
            row_counts=_arc_row_counts(arcs),
            resident_bytes=arc_bytes,
            index_presence=index_presence,
        ),
        families=FamilyTablesManifest(
            row_counts=_family_row_counts(families), resident_bytes=family_bytes
        ),
        ingestion=ingestion,
    )
    return EncodedBuild(manifest=manifest, files=files)


def _checked_un_bytes(files: Mapping[str, bytes], un_mirror: UnMirror, *, label: str) -> int:
    """The size in memory of the UN tables `files` decode to, refusing files
    that do not decode to `un_mirror`."""
    decoded = decode_un_mirror(files, label=label)
    if decoded != un_mirror:
        raise CatalogueArtifactError(f"{label}: the UN tables do not decode to what was encoded")
    return resident_bytes(decoded)


def _checked_arc_bytes(files: Mapping[str, bytes], arcs: ArcTables, *, label: str) -> int:
    """The size in memory of the arc tables `files` decode to, refusing files
    that do not decode to `arcs`."""
    decoded = decode_arc_tables(files, label=label)
    if decoded != arcs:
        raise CatalogueArtifactError(f"{label}: the arc tables do not decode to what was encoded")
    return resident_bytes(decoded)


def _checked_family_bytes(files: Mapping[str, bytes], families: FamilyTables, *, label: str) -> int:
    """The size in memory of the family tables `files` decode to, refusing a file that
    does not decode to `families`."""
    decoded = decode_family_tables(files, label=label)
    if decoded != families:
        raise CatalogueArtifactError(
            f"{label}: the family tables do not decode to what was encoded"
        )
    return resident_bytes(decoded)


def _check_arc_scope(tables: CatalogueTables, arcs: ArcTables, *, label: str) -> None:
    """Refuses arc tables that disagree with the variables the build lists.

    A `dfo` node with arcs must be a listed DFO variable, and an empty one a node the
    build does not list: the `dfo` table holds empty each resolver-index subject the graph
    has no node for (`origin_catalogue.arc_reads.read_catalogue_arcs`), which routes as a
    confirmed absence. The `base` table holds the listed base variables, each DFO variable
    whose node is base Data Commons', such as `Count_Person`, and each base node a
    resolver index names, so a `base` node may be any node admission admits. A `dfo` or
    `base` node must not be one admission refuses, a withheld variable that is not DFO's,
    whose arcs would reach the family summaries. A `schema` node must not be listed. Every
    listed variable must be held by the `dfo` or `base` table, so none reads live."""
    listed = tables.dfo_variables | tables.base_variables.keys()
    refused = tables.withheld_variables - tables.dfo_variables
    for name, outside in (
        (
            "dfo",
            sorted(
                node
                for node, rows in arcs.dfo.items()
                if (node not in tables.dfo_variables if rows else node in listed) or node in refused
            ),
        ),
        ("base", sorted(arcs.base.keys() & refused)),
        ("schema", sorted(arcs.schema.keys() & listed)),
    ):
        if outside:
            raise CatalogueArtifactError(
                f"{label}: {len(outside)} node(s) of the {name} arc table are outside its "
                f"scope, e.g. {outside[:3]!r}"
            )
    if unheld := sorted(listed - arcs.dfo.keys() - arcs.base.keys()):
        raise CatalogueArtifactError(
            f"{label}: the dfo and base arc tables hold none of {len(unheld)} listed "
            f"variable(s), e.g. {unheld[:3]!r}"
        )


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
            "licence": {
                "type": row.licence.type,
                "label": row.licence.label,
                "url": row.licence.url,
            },
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


def _encode_source_documentation(documentation: Mapping[str, SourceDocumentation]) -> bytes:
    rows = [
        {
            "ref": ref,
            "provenances": list(entry.provenances),
            "publisher": entry.publisher,
            "source_data_url": entry.source_data_url,
            "citation": entry.citation,
            "license_url": entry.license_url,
            "last_update": entry.last_update,
            "next_expected_update": entry.next_expected_update,
            "documents": {prop: list(values) for prop, values in entry.documents.items()},
        }
        for ref, entry in sorted(documentation.items())
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
        UN_ARCS_FILE: _encode_arc_targets(list(rows)),
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


def _encode_arc_tables(arcs: ArcTables) -> dict[str, bytes]:
    """The arc tables' four files. Target rows are numbered in first-use
    order over the DFO, base and schema nodes, each table sorted by dcid, so
    equal tables encode to equal bytes."""
    rows: dict[UnArc, int] = {}
    encoded: dict[str, bytes] = {}
    for name, table in (
        (DFO_ARCS_FILE, arcs.dfo),
        (BASE_ARCS_FILE, arcs.base),
        (SCHEMA_ARCS_FILE, arcs.schema),
    ):
        nodes = sorted(table)
        arc_lists = [[rows.setdefault(arc, len(rows)) for arc in table[node]] for node in nodes]
        encoded[name] = _encode_parquet(
            pa.table({"node": nodes, "arcs": arc_lists}, schema=_ARC_ROWS_SCHEMA)
        )
    encoded[ARC_TARGETS_FILE] = _encode_arc_targets(list(rows), labels=arcs.labels)
    return encoded


def _encode_arc_targets(
    rows: list[UnArc], *, labels: Mapping[str, str | None] | None = None
) -> bytes:
    """Target rows, with a `label` column from `labels` when given."""
    columns: dict[str, list[Any]] = {
        "property": [arc.property for arc in rows],
        "dcid": [arc.dcid for arc in rows],
        "value": [arc.value for arc in rows],
        "name": [arc.name for arc in rows],
        "types": [list(arc.types) for arc in rows],
    }
    if labels is None:
        return _encode_parquet(pa.table(columns, schema=_ARC_TARGETS_SCHEMA))
    columns["label"] = [None if arc.dcid is None else labels[arc.dcid] for arc in rows]
    return _encode_parquet(pa.table(columns, schema=_ARC_TABLE_TARGETS_SCHEMA))


def _encode_families(families: FamilyTables) -> bytes:
    """The family table file: the arc tables' rows, then the UN tables', each in key
    order, so equal tables encode to equal bytes."""
    rows = [
        (un, key, summary)
        for un, table in ((False, families.arcs), (True, families.un))
        for key, summary in sorted(table.items(), key=lambda item: _key_order(item[0]))
    ]
    return _encode_parquet(
        pa.table(
            {
                "un": [un for un, _, _ in rows],
                **{
                    name: [key[at] for _, key, _ in rows]
                    for at, name in enumerate(FamilyKey._fields)
                },
                "members": [list(summary.members) for _, _, summary in rows],
                "properties": [list(summary.values) for _, _, summary in rows],
                "values": [
                    [list(found) for found in summary.values.values()] for _, _, summary in rows
                ],
                "shared": [list(summary.shared) for _, _, summary in rows],
                "total": [summary.total for _, _, summary in rows],
                "total_name": [summary.total_name for _, _, summary in rows],
            },
            schema=_FAMILIES_SCHEMA,
        )
    )


def _key_order(key: FamilyKey) -> tuple[tuple[bool, str], ...]:
    """A total order over family keys, a missing qualifier or denominator first."""
    return tuple((field is not None, str(field or "")) for field in key)


def _family_row_counts(families: FamilyTables) -> dict[str, int]:
    return {"families": len(families.arcs), "un_families": len(families.un)}


def _arc_row_counts(arcs: ArcTables) -> dict[str, int]:
    tables = {"dfo_arcs": arcs.dfo, "base_arcs": arcs.base, "schema_arcs": arcs.schema}
    return {
        **{name: len(table) for name, table in tables.items()},
        **{
            f"{name}_empty": sum(not row for row in table.values())
            for name, table in tables.items()
        },
        "arc_targets": len(
            {arc for table in tables.values() for row in table.values() for arc in row}
        ),
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
        "source_documentation": len(tables.source_documentation),
        "superseded_variables": len(tables.superseded),
        "successor_variables": sum(
            successor is not None for successor in tables.superseded.values()
        ),
    }


def _bitset_width(places: int) -> int:
    return (places + 7) // 8


def resident_bytes(tables: CatalogueTables | UnMirror | ArcTables | FamilyTables) -> int:
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
    disagree with the manifest's, a file whose SHA-256 the manifest does not
    record, arc tables whose row counts disagree with the manifest's, that hold
    a node outside their scope or that hold no row for a listed variable
    (`_check_arc_scope`), tables breaking
    `CatalogueTables`', `UnMirror`'s or `ArcTables`' invariants, family
    tables `decode_family_tables` refuses or whose row counts disagree with the
    manifest's, and a successor the UN tables do not admit. Refuses with
    `CatalogueTooLargeError`, before reading any table, a build whose
    `resident_bytes`, `un_mirror.resident_bytes`, `arc_tables.resident_bytes`
    and `families.resident_bytes` together exceed `max_resident_bytes`.

    The snapshot also summarises the base arc table alone (`CatalogueSnapshot.base_families`),
    which `max_resident_bytes` does not count.
    """
    manifest, tables = load_tables(base_uri, max_resident_bytes=max_resident_bytes)
    prefix = join_uri(format_root(base_uri), VERSIONS_PREFIX, manifest.build_id)
    mirror = _read_un_mirror(prefix, manifest)
    arcs = _read_arc_tables(prefix, manifest)
    families = _read_family_tables(prefix, manifest)
    _check_arc_scope(tables, arcs, label=f"catalogue build {manifest.build_id!r} at {prefix!r}")
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
        arcs=arcs,
        families=families,
        un_system_enabled=un_system_enabled,
        # A graph release answers the DFO half of each family, which joins these.
        base_families=summarise_table(arcs.base),
    )


def load_tables(
    base_uri: str, *, max_resident_bytes: int | None = None
) -> tuple[CatalogueManifest, CatalogueTables]:
    """The manifest and tables of the build `current.json` names, without
    its UN tables, which `read_un_mirror` reads, and its arc tables, which
    `read_arc_tables` reads. Refuses what
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
    needed = (
        manifest.resident_bytes
        + manifest.un_mirror.resident_bytes
        + manifest.arc_tables.resident_bytes
        + manifest.families.resident_bytes
    )
    if max_resident_bytes is not None and needed > max_resident_bytes:
        raise CatalogueTooLargeError(
            f"catalogue build {manifest.build_id!r} needs {needed} bytes in memory "
            f"({manifest.un_mirror.resident_bytes} of them for the UN tables, "
            f"{manifest.arc_tables.resident_bytes} for the arc tables and "
            f"{manifest.families.resident_bytes} for the family tables), over the "
            f"{max_resident_bytes}-byte budget"
        )
    files = _read_files(prefix, manifest, _FILES)
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
    files = _read_files(prefix, manifest, _UN_MIRROR_FILES)
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


@dataclass(frozen=True, slots=True)
class BuildUnTables:
    """One build directory's UN rows, the places the UN front door refused
    it, and its UN tables, with the build's id and format."""

    build_id: str
    format_version: int
    rows: Mapping[str, ProvenanceRow]
    blocked_places: frozenset[str]
    mirror: UnMirror


def read_build_un_tables(build_uri: str) -> BuildUnTables:
    """Reads the UN rows, refused places and UN tables of the build
    directory at `build_uri`, whatever its format: the manifest is read as
    JSON for its build id, file names and UN counts alone, with no format
    check. `un.json` and the five UN table files have kept their encoding
    since format 3, and so have the UN rows of `provenances.json`, apart from
    the empty `licence` format 4 added to every row, which a UN row decodes
    to whether or not its build wrote one. Refuses, with
    `CatalogueArtifactError`, a missing or malformed file, UN tables breaking
    `UnMirror`'s invariants or holding a variable no agency's id prefix
    starts, and UN tables whose row counts or agency counts disagree with the
    manifest's."""
    uri = join_uri(build_uri, MANIFEST_FILE)
    try:
        payload = json.loads(_read(uri))
        build_id = _string(payload["build_id"])
        format_version = int(payload["format_version"])
        files = {str(logical): str(name) for logical, name in dict(payload["files"]).items()}
        recorded = UnMirrorManifest(
            row_counts={
                str(k): int(v) for k, v in dict(payload["un_mirror"]["row_counts"]).items()
            },
            agencies={
                str(segment): agency_counts_from_json(counts)
                for segment, counts in dict(payload["un_mirror"]["agencies"]).items()
            },
        )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise CatalogueArtifactError(f"manifest at {uri!r} is malformed: {exc!r}") from exc

    def read(logical: str) -> bytes:
        if logical not in files:
            raise CatalogueArtifactError(f"manifest at {uri!r} names no {logical!r} file")
        return _read(join_uri(build_uri, files[logical]))

    label = f"catalogue build {build_id!r} at {build_uri!r}"
    try:
        un_rows = _provenance_rows(
            {**item, "licence": item.get("licence", _NO_LICENCE)}
            for item in json.loads(read("provenances"))
            if item["data_commons"] == UN_SYSTEM.id
        )
        blocked = frozenset(_string(place) for place in json.loads(read("un"))["blocked_places"])
    except (ValueError, KeyError, TypeError) as exc:
        raise CatalogueArtifactError(f"{label}: {exc!r}") from exc
    mirror = decode_un_mirror(
        {name: read(logical) for logical, name in _UN_MIRROR_FILES.items()}, label=label
    )
    if foreign := foreign_variables(mirror):
        raise CatalogueArtifactError(
            f"{label} holds {len(foreign)} UN variable(s) no agency's id prefix starts, "
            f"e.g. {foreign[:3]!r}"
        )
    if _un_row_counts(mirror) != dict(recorded.row_counts) or mirror.agencies != recorded.agencies:
        raise CatalogueArtifactError(
            f"{label} has UN row counts {_un_row_counts(mirror)!r} and agencies "
            f"{dict(mirror.agencies)!r}, but its manifest records {recorded!r}"
        )
    return BuildUnTables(
        build_id=build_id,
        format_version=format_version,
        rows=un_rows,
        blocked_places=blocked,
        mirror=mirror,
    )


def read_arc_tables(base_uri: str, manifest: CatalogueManifest) -> ArcTables:
    """Reads the arc tables of `manifest`'s build. Refuses, with
    `CatalogueArtifactError`, a missing or malformed file, tables breaking
    `ArcTables`' invariants, and tables whose row counts disagree with the
    manifest's."""
    return _read_arc_tables(
        join_uri(format_root(base_uri), VERSIONS_PREFIX, manifest.build_id), manifest
    )


def _read_arc_tables(prefix: str, manifest: CatalogueManifest) -> ArcTables:
    files = _read_files(prefix, manifest, _ARC_TABLES_FILES)
    label = f"catalogue build {manifest.build_id!r} at {prefix!r}"
    arcs = decode_arc_tables(files, label=label)
    counts = _arc_row_counts(arcs)
    if counts != dict(manifest.arc_tables.row_counts):
        raise CatalogueArtifactError(
            f"{label} has arc table row counts {counts!r}, but its manifest records "
            f"{dict(manifest.arc_tables.row_counts)!r}"
        )
    return arcs


def _read_family_tables(prefix: str, manifest: CatalogueManifest) -> FamilyTables:
    files = _read_files(prefix, manifest, _FAMILY_FILES)
    label = f"catalogue build {manifest.build_id!r} at {prefix!r}"
    families = decode_family_tables(files, label=label)
    counts = _family_row_counts(families)
    if counts != dict(manifest.families.row_counts):
        raise CatalogueArtifactError(
            f"{label} has family row counts {counts!r}, but its manifest records "
            f"{dict(manifest.families.row_counts)!r}"
        )
    return families


def decode_family_tables(files: Mapping[str, bytes], *, label: str) -> FamilyTables:
    """Decodes the family table file, keyed by its standard file name. Rows share one
    string per distinct property, value and key field. Refuses a key twice in one table, a
    family of no member or naming a member twice, a property listed twice or with no value or a value twice, a
    `properties` and `values` of different lengths, a shared property not listed or listed
    with several values, a total without a name or a name without a total, and nulls."""
    try:
        strings: dict[str, str] = {}
        tables: tuple[dict[FamilyKey, FamilySummary], dict[FamilyKey, FamilySummary]] = ({}, {})
        for (
            un,
            population_type,
            measured_property,
            stat_type,
            qualifier,
            denominator,
            unresolved,
            role_bearing,
            members,
            properties,
            values,
            shared,
            total,
            total_name,
        ) in _rows(files[FAMILIES_FILE], _FAMILIES_SCHEMA):
            key = FamilyKey(
                population_type=strings.setdefault(population_type, population_type),
                measured_property=strings.setdefault(measured_property, measured_property),
                stat_type=strings.setdefault(stat_type, stat_type),
                measurement_qualifier=qualifier and strings.setdefault(qualifier, qualifier),
                measurement_denominator=denominator
                and strings.setdefault(denominator, denominator),
                denominator_unresolved=unresolved,
                role_bearing=role_bearing,
            )
            table = tables[un]
            if key in table:
                raise CatalogueTableError(f"family {key!r} appears twice")
            if not members or len(set(members)) != len(members):
                raise CatalogueTableError(f"family {key!r} lists its members malformed")
            if len(properties) != len(values) or len(set(properties)) != len(properties):
                raise CatalogueTableError(f"family {key!r} lists its properties malformed")
            if any(not found or len(set(found)) != len(found) for found in values):
                raise CatalogueTableError(f"family {key!r} lists a property's values malformed")
            by_property = {
                strings.setdefault(prop, prop): tuple(strings.setdefault(v, v) for v in found)
                for prop, found in zip(properties, values, strict=True)
            }
            if any(len(by_property.get(prop, ())) != 1 for prop in shared):
                raise CatalogueTableError(f"family {key!r} shares a property it lists malformed")
            if (total is None) != (total_name is None):
                raise CatalogueTableError(f"family {key!r} has a total without a name")
            table[key] = FamilySummary(
                members=tuple(members),
                values=by_property,
                shared={strings[prop]: by_property[prop][0] for prop in shared},
                total=total,
                total_name=total_name,
                un_only=un,
            )
        return FamilyTables(arcs=tables[0], un=tables[1])
    except (ValueError, KeyError, TypeError, pa.ArrowException) as exc:
        # CatalogueTableError is a ValueError.
        raise CatalogueArtifactError(f"{label}: {exc!r}") from exc


def decode_arc_tables(files: Mapping[str, bytes], *, label: str) -> ArcTables:
    """Decodes the arc table files, keyed by their standard file names.
    Nodes share one `UnArc` per target row and one string per property and
    type, so the tables' memory scales with their distinct values. Refuses
    a missing `label` column, a node twice in one table, a row number out
    of range, two rows naming one node with different labels, and nulls."""
    try:
        strings: dict[str, str] = {}
        targets: list[UnArc] = []
        labels: dict[str, str | None] = {}
        for property, dcid, value, name, kinds, target_label in _rows(
            files[ARC_TARGETS_FILE], _ARC_TABLE_TARGETS_SCHEMA
        ):
            targets.append(
                UnArc(
                    property=strings.setdefault(property, property),
                    dcid=dcid,
                    value=value,
                    name=name,
                    types=tuple(strings.setdefault(kind, kind) for kind in kinds),
                )
            )
            # A label equal to the row's name shares its string.
            target_label = name if target_label == name else target_label
            if dcid is not None and labels.setdefault(dcid, target_label) != target_label:
                raise CatalogueTableError(f"arc target {dcid!r} has two labels")
        tables: list[dict[str, tuple[UnArc, ...]]] = []
        for file_name in (DFO_ARCS_FILE, BASE_ARCS_FILE, SCHEMA_ARCS_FILE):
            table: dict[str, tuple[UnArc, ...]] = {}
            for node, rows in _rows(files[file_name], _ARC_ROWS_SCHEMA):
                if node in table:
                    raise CatalogueTableError(f"node {node!r} appears twice in {file_name}")
                if any(not 0 <= row < len(targets) for row in rows):
                    raise CatalogueTableError(f"node {node!r} names an arc row out of range")
                table[node] = tuple(targets[row] for row in rows)
            tables.append(table)
        dfo, base, schema = tables
        return ArcTables(dfo=dfo, base=base, schema=schema, labels=labels)
    except (ValueError, KeyError, TypeError, pa.ArrowException) as exc:
        # CatalogueTableError and ArcTablesError are ValueErrors.
        raise CatalogueArtifactError(f"{label}: {exc!r}") from exc


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
            for property, dcid, value, name, kinds in _rows(
                files[UN_ARCS_FILE], _ARC_TARGETS_SCHEMA
            )
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


def _read_files(
    prefix: str, manifest: CatalogueManifest, logical: Mapping[str, str]
) -> dict[str, bytes]:
    """The files `logical` names, each logical name mapped to its standard file name, read
    from where the manifest puts them and keyed by standard name. Refuses a file whose
    SHA-256 the manifest does not record or whose bytes do not match it."""
    files: dict[str, bytes] = {}
    for logical_name, name in logical.items():
        uri = join_uri(prefix, _manifest_file(manifest, logical_name))
        raw = _read(uri)
        recorded = manifest.file_sha256.get(logical_name)
        if recorded is None or hashlib.sha256(raw).hexdigest() != recorded:
            raise CatalogueArtifactError(
                f"{uri!r} does not match the SHA-256 the manifest of build "
                f"{manifest.build_id!r} records for {logical_name!r} ({recorded!r})"
            )
        files[name] = raw
    return files


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
            source_documentation=_decode_source_documentation(files[SOURCE_DOCUMENTATION_FILE]),
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


_NO_LICENCE: Final[Mapping[str, None]] = {"type": None, "label": None, "url": None}
"""The `licence` of a row that states none, as `provenances.json` encodes it."""


def _decode_provenances(raw: bytes) -> dict[str, ProvenanceRow]:
    return _provenance_rows(json.loads(raw))


def _provenance_rows(items: Iterable[Mapping[str, Any]]) -> dict[str, ProvenanceRow]:
    rows: dict[str, ProvenanceRow] = {}
    for item in items:
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
            licence=Licence(
                type=_optional_string(item["licence"]["type"]),
                label=_optional_string(item["licence"]["label"]),
                url=_optional_string(item["licence"]["url"]),
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


def _decode_source_documentation(raw: bytes) -> dict[str, SourceDocumentation]:
    documentation: dict[str, SourceDocumentation] = {}
    for item in json.loads(raw):
        ref = _string(item["ref"])
        if ref in documentation:
            raise CatalogueTableError(f"source documentation for {ref!r} appears twice")
        documentation[ref] = SourceDocumentation(
            provenances=tuple(_string(provenance) for provenance in item["provenances"]),
            publisher=_optional_string(item["publisher"]),
            source_data_url=_optional_string(item["source_data_url"]),
            citation=_optional_string(item["citation"]),
            license_url=_optional_string(item["license_url"]),
            last_update=_optional_string(item["last_update"]),
            next_expected_update=_optional_string(item["next_expected_update"]),
            documents={
                _string(prop): tuple(_string(value) for value in values)
                for prop, values in dict(item["documents"]).items()
            },
        )
    return documentation


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
    "BuildUnTables",
    "CatalogueArtifactError",
    "CatalogueTooLargeError",
    "EncodedBuild",
    "decode_arc_tables",
    "decode_family_tables",
    "decode_un_mirror",
    "encode_build",
    "format_root",
    "load_catalogue",
    "load_tables",
    "publish_build",
    "read_arc_tables",
    "read_build_un_tables",
    "read_manifest",
    "read_published_pointer",
    "read_un_mirror",
    "resident_bytes",
]
