"""Local-file serializers for coverage artifacts.

URI handling lives in `series_coverage.loader` and `artifact_publication.uri`.
`read_manifest` and `read_current` support artifact tests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from artifact_publication.jobs import Ingestion
from series_coverage.schema import (
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)

# The table-size estimate assumes pyarrow's default zstd level.
_PARQUET_COMPRESSION = "zstd"


def write_coverage_table(table: pa.Table, path: Path, *, horizon: YearHorizon) -> None:
    """Checks `table` against `horizon.schema`, then writes zstd Parquet.

    A mismatch raises `pa.ArrowInvalid`.
    """
    if not table.schema.equals(horizon.schema):
        raise pa.ArrowInvalid(
            f"coverage table schema {table.schema!r} does not match {horizon}'s "
            f"schema {horizon.schema!r}"
        )
    pq.write_table(table, path, compression=_PARQUET_COMPRESSION)


def write_role_names(role_names: dict[str, RoleNames], path: Path) -> None:
    """Writes `role_names` (variable dcid -> `RoleNames`) as JSON: `{variable:
    {"donor": ..., "recipient": ...}}`.
    """
    payload = {
        variable: {"donor": names.donor, "recipient": names.recipient}
        for variable, names in role_names.items()
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_facet_metadata(metadata: Mapping[str, FacetMetadata], path: Path) -> None:
    """Writes `metadata` (facet_id -> `FacetMetadata`) as JSON: `{facet_id:
    {"provenance_id": ..., "measurement_method": ..., "unit": ...,
    "scaling_factor": ..., "observation_period": ...}}`, `null` for a field
    the job couldn't resolve.
    """
    payload = {
        facet_id: {
            "provenance_id": meta.provenance_id,
            "measurement_method": meta.measurement_method,
            "unit": meta.unit,
            "scaling_factor": meta.scaling_factor,
            "observation_period": meta.observation_period,
        }
        for facet_id, meta in metadata.items()
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_manifest(manifest: CoverageManifest, path: Path) -> None:
    payload = {
        "stamp": manifest.stamp,
        "build_id": manifest.build_id,
        "built_at": manifest.built_at,
        "source": manifest.source,
        "row_counts": manifest.row_counts,
        "files": manifest.files,
        "epoch_year": manifest.horizon.epoch_year,
        "mask_words": manifest.horizon.mask_words,
        "ingestion": None if manifest.ingestion is None else manifest.ingestion.to_json(),
        "format_version": manifest.format_version,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def read_manifest(path: Path) -> CoverageManifest:
    """Raises `KeyError` naming the missing field for a manifest missing
    any of `stamp`/`build_id`/`built_at`/`source`/`row_counts`/`files`/
    `epoch_year`/`mask_words`/`format_version`. A manifest without
    `ingestion` reads as `None`.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
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
            format_version=int(payload["format_version"]),
        )
    except KeyError as exc:
        raise KeyError(f"manifest at {path} is missing {exc}") from exc


def write_current(pointer: CoveragePointer, path: Path) -> None:
    """Writes the `current.json` pointer (`CoveragePointer.to_json`) to
    `path`, which belongs at the format root beside `versions/`. A
    publisher writes it last, after the build directory it names is
    complete, so the pointer is the commit point.
    """
    path.write_text(pointer.to_json(), encoding="utf-8")


def read_current(path: Path) -> CoveragePointer:
    """Reads a `current.json` pointer back. Raises `ValueError` for a
    pointer `CoveragePointer.from_json` refuses.
    """
    return CoveragePointer.from_json(path.read_text(encoding="utf-8"))


__all__ = [
    "read_current",
    "read_manifest",
    "write_coverage_table",
    "write_current",
    "write_facet_metadata",
    "write_manifest",
    "write_role_names",
]
