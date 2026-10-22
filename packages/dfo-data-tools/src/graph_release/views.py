"""The serving views of a release: the `CoverageIndex` and the code and measure indexes.

The builder's gate (`graph_release.gate.check_views`) and a serving process build them through
these functions, so a release the gate passes is one a serving process can open. Each raises
`ViewRefused` when its view does not build.

`coverage_index` builds the coverage rows of `series` under the role map `edges` and the
declared role axes give (`graph_release.coverage`), takes each covered facet's metadata from
`facets`, and checks the rows as `series_coverage.loader` checks a published coverage build.
`code_indexes` parses the `code_index_blocks` document with `code_index.loader`.

For a published release, `release_coverage_index` takes the role axes from the `declarations`
part and the build year from the manifest's `read_timestamp`. `release_code_indexes` stamps the
document with the release id as its `coverage_build`, since the coverage it joined is the
release's own, and the read time as its `exported_at`. The part cannot carry either: its bytes
feed the release id, and a stamp that moved with every build would give an unchanged index a
new part in every release. Its `content_hash` covers the blocks alone, so it moves only when
the index does.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final, cast

import pyarrow as pa
import pyarrow.compute as pc

from code_index.loader import parse_index, parse_measure_axes
from code_index.query import CodeIndex
from code_index.types import MeasureIndex
from dfo_platform.roles import RoleAxis
from graph_release.coverage import HORIZON, coverage_rows, role_map
from graph_release.format import LoadedRelease, Manifest
from graph_release.platform import PlatformRefusal
from series_coverage.index import CoverageIndex
from series_coverage.loader import CoverageArtifactError, checked_index
from series_coverage.schema import FACET_COLUMN, FacetMetadata

CODE_INDEX_PART: Final[str] = "code_index_blocks"
DECLARATIONS_PART: Final[str] = "declarations"

_EXPORTED_AT_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"


class ViewRefused(Exception):
    """A release whose serving view does not build."""


def coverage_index(
    *,
    series: pa.Table,
    edges: pa.Table,
    facets: pa.Table,
    role_axes: Mapping[str, RoleAxis],
    build_year: int | None,
) -> CoverageIndex:
    """The `CoverageIndex` of the `series`, `edges` and `facets` parts under `role_axes`.

    Raises `ViewRefused` where `role_map` or `coverage_rows` refuses, or where the rows fail a
    check of `series_coverage.loader.checked_index`.
    """
    try:
        roles = role_map(edges, role_axes)
        rows = coverage_rows(series, roles)
        return checked_index(
            rows,
            roles,
            _facet_metadata(facets, rows),
            horizon=HORIZON,
            build_year=build_year,
            label="the release's coverage rows",
        )
    except (PlatformRefusal, CoverageArtifactError, ValueError) as exc:
        raise ViewRefused(f"the coverage view does not build: {exc}") from exc


def _facet_metadata(facets: pa.Table, rows: pa.Table) -> dict[str, FacetMetadata]:
    """Each facet of `rows` with its `facets` row's metadata. `scaling_factor` stays `None`,
    as the coverage build's `facets.json` holds it, since `TimeSeries` carries none."""
    covered = rows[FACET_COLUMN].cast(pa.string()).unique()
    held = facets.filter(pc.is_in(facets["facet_id"], value_set=covered))
    return {
        row["facet_id"]: FacetMetadata(
            provenance_id=row["provenance"],
            measurement_method=row["measurement_method"],
            unit=row["unit"],
            observation_period=row["observation_period"],
        )
        for row in held.select(
            ["facet_id", "provenance", "measurement_method", "unit", "observation_period"]
        ).to_pylist()
    }


def code_indexes(document: Mapping[str, object]) -> tuple[CodeIndex, MeasureIndex]:
    """The code index and measure index of a code index `document`.

    Raises `ViewRefused` where `parse_index` or `parse_measure_axes` refuses it.
    """
    try:
        return parse_index(document, source=CODE_INDEX_PART), parse_measure_axes(
            document, source=CODE_INDEX_PART
        )
    except ValueError as exc:
        raise ViewRefused(f"the code index does not load: {exc}") from exc


def read_document(data: bytes) -> dict[str, object]:
    """The gzipped JSON object of a `code_index_blocks` part. Raises `ViewRefused` for bytes
    that are not one."""
    try:
        document = json.loads(gzip.decompress(data))
    except (OSError, EOFError, ValueError) as exc:
        raise ViewRefused(f"the {CODE_INDEX_PART} part is not gzipped JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ViewRefused(f"the {CODE_INDEX_PART} part is not a JSON object")
    return document


def declared_role_axes(data: bytes) -> Mapping[str, RoleAxis]:
    """The `role_axes` of a `declarations` part. Raises `ViewRefused` for a part that holds
    none."""
    try:
        role_axes = json.loads(data)["role_axes"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ViewRefused(f"the {DECLARATIONS_PART} part holds no role_axes: {exc!r}") from exc
    if not isinstance(role_axes, dict):
        raise ViewRefused(f"the {DECLARATIONS_PART} part's role_axes is not an object")
    return cast("Mapping[str, RoleAxis]", role_axes)


def release_coverage_index(loaded: LoadedRelease) -> CoverageIndex:
    """`coverage_index` over a published release, with the role axes of its `declarations`
    part and the year of its `read_timestamp` as the build year. Raises `ViewRefused` for a
    release that lacks a part the view reads."""
    try:
        series, edges, facets = (loaded.tables[name] for name in ("series", "edges", "facets"))
        declarations = loaded.blobs[DECLARATIONS_PART]
    except KeyError as exc:
        raise ViewRefused(f"the release holds no {exc.args[0]} part") from exc
    return coverage_index(
        series=series,
        edges=edges,
        facets=facets,
        role_axes=declared_role_axes(declarations),
        build_year=_read_at(loaded.manifest).year,
    )


def release_code_indexes(loaded: LoadedRelease) -> tuple[CodeIndex, MeasureIndex]:
    """`code_indexes` of a published release's `code_index_blocks` part, its stamp naming the
    release id as `coverage_build` and the manifest's `read_timestamp` as `exported_at`. Raises
    `ViewRefused` for a release that lacks the part."""
    data = loaded.blobs.get(CODE_INDEX_PART)
    if data is None:
        raise ViewRefused(f"the release holds no {CODE_INDEX_PART} part")
    document = read_document(data)
    if isinstance(stamp := document.get("stamp"), dict):
        document["stamp"] = {
            **stamp,
            "coverage_build": loaded.manifest.release_id,
            "exported_at": _read_at(loaded.manifest).strftime(_EXPORTED_AT_FORMAT),
        }
    return code_indexes(document)


def _read_at(manifest: Manifest) -> datetime:
    return datetime.fromisoformat(manifest.read_timestamp).astimezone(UTC)


__all__ = [
    "CODE_INDEX_PART",
    "DECLARATIONS_PART",
    "ViewRefused",
    "code_indexes",
    "coverage_index",
    "declared_role_axes",
    "read_document",
    "release_code_indexes",
    "release_coverage_index",
]
