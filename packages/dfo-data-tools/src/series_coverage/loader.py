"""Loads one immutable coverage build from a `gs://` or `file://` URI.

`current.json`, or another pointer in its form, selects a build directory. Its
manifest names the Parquet coverage table, role names, and facet metadata.
`series_coverage.shared` loads the result at surface construction and manages
refreshes.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from artifact_publication.jobs import Ingestion
from artifact_publication.uri import join_uri, read_uri_bytes
from series_coverage._bits import popcount64
from series_coverage.index import CoverageIndex
from series_coverage.schema import (
    ABSENT_ENTITY,
    COVERAGE_FORMAT_VERSION,
    DONOR_COLUMN,
    FACET_COLUMN,
    FORMAT_ROOT,
    N_OBS_COLUMN,
    POINTER_FILE,
    RECIPIENT_COLUMN,
    VARIABLE_COLUMN,
    VERSIONS_PREFIX,
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)


def format_root(base_uri: str) -> str:
    """This image's format root under `base_uri`: the only prefix the
    loader, the job, and verify read or write."""
    return join_uri(base_uri, FORMAT_ROOT)


# The series key columns are read dictionary-encoded, as `CoverageIndex`
# holds them: decoding them to plain strings first would allocate one
# string per row for each of the four columns only for the index to
# re-encode them.
_KEY_COLUMNS = (VARIABLE_COLUMN, FACET_COLUMN, DONOR_COLUMN, RECIPIENT_COLUMN)


def _resident_schema(horizon: YearHorizon) -> pa.Schema:
    """`horizon.schema` with the series key columns dictionary-typed."""
    return pa.schema(
        [
            field.with_type(pa.dictionary(pa.int32(), field.type))
            if field.name in _KEY_COLUMNS
            else field
            for field in horizon.schema
        ]
    )


class CoverageArtifactError(Exception):
    """Raised when a published coverage artifact is missing or malformed.

    Checks include pointer/manifest agreement, format and schema, mask
    popcounts, unique series keys, and complete facet metadata. Surface
    construction propagates this error.
    """


@dataclass(frozen=True, slots=True)
class LoadedCoverage:
    manifest: CoverageManifest
    index: CoverageIndex


def load_coverage(base_uri: str, *, pointer: CoveragePointer | None = None) -> LoadedCoverage:
    """Loads the build `pointer` names, or the one `current.json` names when it is `None`:
    the manifest, table, role names, and facet metadata, then builds a `CoverageIndex` with
    the manifest's horizon. It validates format, schema, row-mask counts, unique series keys,
    and facet metadata.

    Raises `CoverageArtifactError` with the failed check or missing file.
    """
    root = format_root(base_uri)
    if pointer is None:
        pointer = _read_pointer(root)
    prefix = join_uri(root, VERSIONS_PREFIX, pointer.build_id)
    manifest = _read_manifest(prefix, root=root, pointer=pointer)
    if (manifest.stamp, manifest.build_id) != (pointer.stamp, pointer.build_id):
        raise CoverageArtifactError(
            f"the pointer at {root!r} names stamp {pointer.stamp!r} build "
            f"{pointer.build_id!r}, but the manifest at {prefix!r} carries stamp "
            f"{manifest.stamp!r} build {manifest.build_id!r}; the two disagree"
        )
    table = _read_table(prefix, manifest)
    role_names = _read_role_names(prefix, manifest)
    facet_metadata = _read_facet_metadata(prefix, manifest)
    label = f"coverage artifact at {prefix!r} (stamp {pointer.stamp!r})"
    _check_popcount_invariant(table, manifest.horizon, label=label)
    _check_facet_metadata_completeness(table, facet_metadata, label=label)
    _check_role_slots(table, role_names, label=label)
    try:
        index = CoverageIndex(table, role_names, facet_metadata, horizon=manifest.horizon)
    except ValueError as exc:
        raise CoverageArtifactError(f"{label}: {exc}") from exc
    return LoadedCoverage(manifest=manifest, index=index)


def _read_pointer(root: str) -> CoveragePointer:
    uri = join_uri(root, POINTER_FILE)
    try:
        raw = read_uri_bytes(uri)
    except FileNotFoundError as exc:
        raise CoverageArtifactError(f"no current.json at {uri!r}: {exc}") from exc
    try:
        return CoveragePointer.from_json(raw)
    except ValueError as exc:
        raise CoverageArtifactError(f"current.json at {uri!r} is malformed: {exc}") from exc


def read_published_pointer(base_uri: str) -> CoveragePointer | None:
    """Reads `current.json` once, returning `None` if it is missing or malformed.

    `CoverageService` uses this cheap check before a full load.
    """
    uri = join_uri(format_root(base_uri), POINTER_FILE)
    try:
        raw = read_uri_bytes(uri)
    except FileNotFoundError:
        return None
    try:
        return CoveragePointer.from_json(raw)
    except ValueError:
        return None


_MANIFEST_FIELDS = (
    "stamp",
    "build_id",
    "built_at",
    "source",
    "row_counts",
    "files",
    "epoch_year",
    "mask_words",
)


def _read_manifest(prefix: str, *, root: str, pointer: CoveragePointer) -> CoverageManifest:
    uri = join_uri(prefix, "manifest.json")
    try:
        raw = read_uri_bytes(uri)
    except FileNotFoundError as exc:
        raise CoverageArtifactError(f"no manifest.json at {uri!r}: {exc}") from exc
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"not a JSON object: {payload!r}")
    except ValueError as exc:
        raise CoverageArtifactError(f"manifest.json at {uri!r} is malformed: {exc}") from exc
    if "format_version" not in payload:
        raise CoverageArtifactError(
            f"manifest.json at {uri!r} is missing required field 'format_version'"
        )
    try:
        format_version = int(payload["format_version"])
    except (ValueError, TypeError) as exc:
        raise CoverageArtifactError(f"manifest.json at {uri!r} is malformed: {exc}") from exc
    if format_version != COVERAGE_FORMAT_VERSION:
        raise CoverageArtifactError(
            f"coverage artifact at {root!r} (build {pointer.build_id!r}) has format_version "
            f"{format_version}, but this code's own COVERAGE_FORMAT_VERSION is "
            f"{COVERAGE_FORMAT_VERSION}; refusing to load a different on-disk format rather "
            "than misinterpret it"
        )
    missing = [field for field in _MANIFEST_FIELDS if field not in payload]
    if missing:
        raise CoverageArtifactError(
            f"manifest.json at {uri!r} is missing required field(s) {missing!r}; it does not "
            f"describe a format {COVERAGE_FORMAT_VERSION} artifact"
        )
    try:
        return CoverageManifest(
            stamp=payload["stamp"],
            build_id=payload["build_id"],
            built_at=payload["built_at"],
            source=payload["source"],
            row_counts=dict(payload["row_counts"]),
            files=dict(payload["files"]),
            horizon=YearHorizon(
                epoch_year=int(payload["epoch_year"]), mask_words=int(payload["mask_words"])
            ),
            ingestion=(
                None
                if (ingestion := payload.get("ingestion")) is None
                else Ingestion.from_json(ingestion)
            ),
            format_version=format_version,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise CoverageArtifactError(f"manifest.json at {uri!r} is malformed: {exc}") from exc


def _manifest_file(manifest: CoverageManifest, *, logical_name: str) -> str:
    try:
        return manifest.files[logical_name]
    except KeyError as exc:
        raise CoverageArtifactError(
            f"manifest.json's files map has no entry for {logical_name!r} (has {sorted(manifest.files)})"
        ) from exc


def _read_table(prefix: str, manifest: CoverageManifest) -> pa.Table:
    relative = _manifest_file(manifest, logical_name="coverage")
    uri = join_uri(prefix, relative)
    try:
        raw = read_uri_bytes(uri)
    except FileNotFoundError as exc:
        raise CoverageArtifactError(
            f"manifest names coverage file {uri!r}, which is missing: {exc}"
        ) from exc
    try:
        table = pq.read_table(io.BytesIO(raw), read_dictionary=list(_KEY_COLUMNS))
        return table.cast(_resident_schema(manifest.horizon))
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
        raise CoverageArtifactError(
            f"coverage Parquet at {uri!r} does not match the manifest's {manifest.horizon} "
            f"schema: {exc}"
        ) from exc


def _read_role_names(prefix: str, manifest: CoverageManifest) -> dict[str, RoleNames]:
    relative = _manifest_file(manifest, logical_name="role_names")
    uri = join_uri(prefix, relative)
    try:
        raw = read_uri_bytes(uri)
    except FileNotFoundError as exc:
        raise CoverageArtifactError(
            f"manifest names role_names file {uri!r}, which is missing: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
        return {variable: _role_names_entry(variable, names) for variable, names in payload.items()}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise CoverageArtifactError(f"role_names.json at {uri!r} is malformed: {exc}") from exc


def _role_names_entry(variable: str, names: dict[str, object]) -> RoleNames:
    """One `role_names.json` entry: a string or null donor and recipient, at least one a
    string."""
    donor, recipient = names["donor"], names["recipient"]
    if not (donor is None or isinstance(donor, str)) or not (
        recipient is None or isinstance(recipient, str)
    ):
        raise ValueError(
            f"variable {variable!r} needs a string or null donor and recipient; got {names!r}"
        )
    try:
        return RoleNames(donor=donor, recipient=recipient)
    except ValueError as exc:
        raise ValueError(f"variable {variable!r}: {exc}; got {names!r}") from exc


def _read_facet_metadata(prefix: str, manifest: CoverageManifest) -> dict[str, FacetMetadata]:
    relative = _manifest_file(manifest, logical_name="facets")
    uri = join_uri(prefix, relative)
    try:
        raw = read_uri_bytes(uri)
    except FileNotFoundError as exc:
        raise CoverageArtifactError(
            f"manifest names facets file {uri!r}, which is missing: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
        return {
            facet_id: FacetMetadata(
                provenance_id=fields.get("provenance_id"),
                measurement_method=fields.get("measurement_method"),
                unit=fields.get("unit"),
                scaling_factor=fields.get("scaling_factor"),
                observation_period=fields.get("observation_period"),
            )
            for facet_id, fields in payload.items()
        }
    except (ValueError, AttributeError) as exc:
        raise CoverageArtifactError(f"facets.json at {uri!r} is malformed: {exc}") from exc


def _check_facet_metadata_completeness(
    table: pa.Table, facet_metadata: dict[str, FacetMetadata], *, label: str
) -> None:
    """Requires metadata for every `facet_id` in `table`.

    Missing entries raise `CoverageArtifactError` before candidate ranking.
    """
    if table.num_rows == 0:
        return
    table_facet_ids = set(cast("list[str]", pc.unique(table[FACET_COLUMN]).to_pylist()))
    missing = sorted(table_facet_ids - facet_metadata.keys())
    if missing:
        raise CoverageArtifactError(
            f"{label} has {len(missing)} "
            f"facet_id(s) with no facets.json entry, e.g. {missing[:10]!r}; the building "
            "job is expected to guarantee an entry for every facet_id before publishing"
        )


def _variables_by_absent_entity(table: pa.Table, column: str) -> tuple[set[str], set[str]]:
    """The variables with at least one row holding `ABSENT_ENTITY` in `column`, and those
    with at least one row holding another value, tested once per dictionary entry."""
    absent_rows = pa.chunked_array(
        [
            pc.take(
                pc.equal(chunk.dictionary, pa.scalar(ABSENT_ENTITY, type=pa.string())),
                chunk.indices,
            )
            for chunk in cast("list[pa.DictionaryArray]", table[column].chunks)
        ],
        type=pa.bool_(),
    )

    def variables(rows: pa.ChunkedArray) -> set[str]:
        found = pc.unique(table[VARIABLE_COLUMN].filter(rows))
        return set(cast("list[str]", found.cast(pa.string()).to_pylist()))

    return variables(absent_rows), variables(pc.invert(absent_rows))


def _check_role_slots(table: pa.Table, role_names: dict[str, RoleNames], *, label: str) -> None:
    """Requires every row's entities to match its variable's `RoleNames`: every row of a
    variable with no donor role holds an empty `donor`, and no row of any other variable
    does, and the same for `recipient`."""
    if table.num_rows == 0:
        return
    covered = set(
        cast("list[str]", pc.unique(table[VARIABLE_COLUMN]).cast(pa.string()).to_pylist())
    )
    named = [(variable, names) for variable in covered if (names := role_names.get(variable))]
    for role, column, lacking in (
        ("donor", DONOR_COLUMN, {v for v, n in named if n.donor is None}),
        ("recipient", RECIPIENT_COLUMN, {v for v, n in named if n.recipient is None}),
    ):
        with_absent, with_entity = _variables_by_absent_entity(table, column)
        mismatched = sorted((lacking & with_entity) | (with_absent - lacking))
        if mismatched:
            raise CoverageArtifactError(
                f"{label} has {len(mismatched)} variable(s) whose rows' {role} column "
                f"disagrees with role_names.json (every row of a variable with a null {role} "
                f"role holds an empty {role}, and no other row does), e.g. {mismatched[:10]!r}"
            )


def _check_popcount_invariant(table: pa.Table, horizon: YearHorizon, *, label: str) -> None:
    """Checks each row's `n_obs` against the total set bits in its mask words."""
    if table.num_rows == 0:
        return
    n_obs = table[N_OBS_COLUMN].to_numpy(zero_copy_only=False).astype(np.uint64)
    bits = np.zeros(table.num_rows, dtype=np.uint64)
    for name in horizon.mask_columns:
        bits += popcount64(table[name].to_numpy(zero_copy_only=False).astype(np.uint64))
    mismatched = n_obs != bits
    count = int(mismatched.sum())
    if count:
        first_index = int(np.flatnonzero(mismatched)[0])
        raise CoverageArtifactError(
            f"{label} has {count} row(s) where "
            f"n_obs != the year-mask bit count, e.g. row {first_index}; the building job is "
            "expected to guarantee this invariant before publishing"
        )


__all__ = [
    "VERSIONS_PREFIX",
    "CoverageArtifactError",
    "LoadedCoverage",
    "format_root",
    "load_coverage",
    "read_published_pointer",
]
