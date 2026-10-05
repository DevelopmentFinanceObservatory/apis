"""Whether a coverage build carries the packaged code index's provenances, and the pointer
that names the newest build that does.

The code index export joins each StatVar's provenances from a coverage build. Dataset election
reads those provenances, and a mint reads facet provenances from the coverage build the service
serves, so the two agree only while the build carries the index's provenances.
`index_coverage_differences` is the one test of that: `python -m series_coverage.verify_index`
runs it at deploy, the coverage job after each publish, and `CoverageService` before it serves
a build.

`<format root>/compatible/<content_hash>.json`, in `current.json`'s form, names a build
compatible with the index whose `IndexStamp.content_hash` it carries. `record_if_compatible`
writes it, and only for a compatible build, so the pointer names the build most recently
checked. The coverage job's prune keeps every build such a pointer names, so a service whose
index the latest build does not match can still load one that does.

`<format root>/serving/code_index.json` holds the `content_hash` of the index every service ran
when the last deploy finished: `python -m series_coverage.record_serving_index` writes it as the
deploy's final step, after both services moved. A pointer written before it belongs to an index
no serving revision runs, unless it is that index's own, so the job expires only such pointers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from artifact_publication.uri import (
    join_uri,
    list_uri_object_times,
    read_uri_bytes,
    write_uri_bytes,
)
from code_index.query import CodeIndex
from series_coverage.loader import CoverageArtifactError, LoadedCoverage, format_root
from series_coverage.schema import COMPATIBLE_PREFIX, SERVING_PREFIX, CoveragePointer

_MAX_VARIABLES_NAMED = 20
_SERVING_INDEX_FILE = "code_index.json"


@dataclass(frozen=True, slots=True)
class IndexCoverageDifferences:
    """Where a code index and a coverage build disagree.

    `variables` maps each differing StatVar to its index provenances and the build's.
    `datasets_without_rows` lists declared datasets no index row holds in the build.
    """

    variables: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]
    datasets_without_rows: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.variables or self.datasets_without_rows)


def index_coverage_differences(
    index: CodeIndex, coverage_provenances: Mapping[str, tuple[str, ...]]
) -> IndexCoverageDifferences:
    """Compare each index row's `provenances` with `coverage_provenances` for its dcid.

    A variable the build does not cover reads as `()`. Variables the build covers and the index
    does not hold are outside the comparison.
    """
    variables: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    populated: set[str] = set()
    for row in index.statvars:
        covered = coverage_provenances.get(row.dcid, ())
        populated.update(covered)
        if covered != row.provenances:
            variables[row.dcid] = (row.provenances, covered)
    return IndexCoverageDifferences(
        variables=variables,
        datasets_without_rows=tuple(sorted(index.datasets.keys() - populated)),
    )


def describe_differences(differences: IndexCoverageDifferences) -> list[str]:
    """One line per declared dataset with no row and per differing variable, naming at most
    twenty variables in dcid order and counting the rest."""
    lines = [
        f"declared dataset {provenance!r} has no row in the build"
        for provenance in differences.datasets_without_rows
    ]
    named = sorted(differences.variables)[:_MAX_VARIABLES_NAMED]
    for dcid in named:
        in_index, in_build = differences.variables[dcid]
        lines.append(f"{dcid}: index {list(in_index)!r}, build {list(in_build)!r}")
    if len(differences.variables) > len(named):
        lines.append(
            f"and {len(differences.variables) - len(named)} more of "
            f"{len(differences.variables)} differing variables"
        )
    return lines


def _content_hash(index: CodeIndex) -> str:
    content_hash = index.stamp.content_hash
    if not content_hash:
        raise ValueError("the code index stamp carries no content_hash to name its pointer by")
    return content_hash


def compatible_pointer_uri(base_uri: str, index: CodeIndex) -> str:
    """The URI of `index`'s compatibility pointer under `base_uri`'s format root.

    Raises `ValueError` for an index whose stamp carries no `content_hash`.
    """
    return join_uri(format_root(base_uri), COMPATIBLE_PREFIX, f"{_content_hash(index)}.json")


def read_compatible_pointer(base_uri: str, index: CodeIndex) -> CoveragePointer | None:
    """The build `index`'s compatibility pointer names, or `None` when it has none.

    Raises `CoverageArtifactError` for a pointer that does not parse.
    """
    uri = compatible_pointer_uri(base_uri, index)
    try:
        raw = read_uri_bytes(uri)
    except FileNotFoundError:
        return None
    try:
        return CoveragePointer.from_json(raw)
    except ValueError as exc:
        raise CoverageArtifactError(f"the pointer at {uri!r} is malformed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ServingIndex:
    """The code index every service ran when the last deploy finished, and when the deploy
    recorded it, in the storage listing's clock."""

    content_hash: str
    recorded_at: float


def record_serving_index(base_uri: str, index: CodeIndex) -> None:
    """Records `index` as the one every service runs.

    Raises `ValueError` for an index whose stamp carries no `content_hash`.
    """
    payload = json.dumps({"content_hash": _content_hash(index)}, sort_keys=True)
    write_uri_bytes(
        join_uri(format_root(base_uri), SERVING_PREFIX, _SERVING_INDEX_FILE),
        payload.encode("utf-8"),
    )


def read_serving_index(base_uri: str) -> ServingIndex | None:
    """The serving index record under `base_uri`'s format root, or `None` when no deploy has
    written one.

    Raises `CoverageArtifactError` for a record that does not parse.
    """
    prefix = join_uri(format_root(base_uri), SERVING_PREFIX)
    recorded_at = list_uri_object_times(prefix).get(_SERVING_INDEX_FILE)
    if recorded_at is None:
        return None
    uri = join_uri(prefix, _SERVING_INDEX_FILE)
    try:
        payload = json.loads(read_uri_bytes(uri))
        content_hash = payload["content_hash"]
    except (ValueError, KeyError, TypeError) as exc:
        raise CoverageArtifactError(f"the serving index at {uri!r} is malformed: {exc}") from exc
    if not isinstance(content_hash, str) or not content_hash:
        raise CoverageArtifactError(
            f"the serving index at {uri!r} has no string 'content_hash': {payload!r}"
        )
    return ServingIndex(content_hash=content_hash, recorded_at=recorded_at)


def record_if_compatible(
    base_uri: str, index: CodeIndex, loaded: LoadedCoverage
) -> IndexCoverageDifferences:
    """Points `index`'s compatibility pointer at `loaded`'s build when the build carries the
    index's provenances, and returns the differences, empty when it recorded the build."""
    differences = index_coverage_differences(index, loaded.index.variable_provenances())
    if not differences:
        pointer = CoveragePointer(stamp=loaded.manifest.stamp, build_id=loaded.manifest.build_id)
        write_uri_bytes(compatible_pointer_uri(base_uri, index), pointer.to_json().encode("utf-8"))
    return differences


__all__ = [
    "IndexCoverageDifferences",
    "ServingIndex",
    "compatible_pointer_uri",
    "describe_differences",
    "index_coverage_differences",
    "read_compatible_pointer",
    "read_serving_index",
    "record_if_compatible",
    "record_serving_index",
]
