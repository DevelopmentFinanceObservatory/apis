"""The shadow diff: a release against the artifacts it will replace, behind `GRAPH_RELEASE_SHADOW`.

`shadow_ready` tells the builder whether both comparators, the published coverage build and the
origin catalogue, were built from the release's import. The builder waits until they were,
since a diff against an earlier import fails for reasons the release did not cause.

`shadow_diff` compares the release's coverage rows and role map with the published coverage
build, and its DFO arcs with the origin catalogue's DFO arc table. A difference is a finding,
except the pinned one, which the report counts: the origin catalogue stores raw
`constraintProperties` rows, so its arcs keep the import-added properties every reader drops
(`dc_graph.visible_arcs.IMPORT_ADDED_CONSTRAINT_PROPERTIES`). A comparator that cannot be read,
or that no longer holds the release's import, is unavailable.

The code index's graph blocks are compared with the committed `index.json.gz` as a report only:
that file is refreshed by hand, so it is not at the release's import. The report marks the
differences that name `PINNED_INDEX_VARIABLES`.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Mapping
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Final, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from google.api_core.exceptions import GoogleAPIError

from artifact_publication.uri import join_uri, read_uri_bytes
from code_index.loader import default_index_path
from dc_graph.visible_arcs import IMPORT_ADDED_CONSTRAINT_PROPERTIES
from dfo_platform.roles import RoleAxis
from graph_release.coverage import coverage_rows, role_map
from graph_release.gate import (
    Dependency,
    GateCode,
    GateFinding,
    GateResult,
    GateUnavailable,
    dfo_variables,
    merge,
    release_arcs,
    visible_edges,
)
from graph_release.platform import PlatformParts, PlatformRefusal
from origin_catalogue.artifact import CatalogueArtifactError, read_arc_tables, read_manifest
from origin_catalogue.artifact import format_root as origin_root
from origin_catalogue.schema import CataloguePointer
from series_coverage.loader import (
    CoverageArtifactError,
    _read_manifest,
    _read_role_names,
    _read_table,
)
from series_coverage.loader import format_root as coverage_root
from series_coverage.schema import (
    DONOR_COLUMN,
    FACET_COLUMN,
    POINTER_FILE,
    RECIPIENT_COLUMN,
    VARIABLE_COLUMN,
    VERSIONS_PREFIX,
    CoveragePointer,
)

PINNED_ARC_PROPERTY: Final[str] = "constraintProperties"
"""The origin catalogue's arcs of this property to an import-added property are pinned."""

PINNED_INDEX_VARIABLES: Final[frozenset[str]] = frozenset(
    {"PrimaryIncome_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal"}
)
"""The variable the release's `listed_statvars` and `source_codes` hold and the committed index
lacks, for a reason not yet traced."""

INDEX_BLOCK_KEYS: Final[Mapping[str, tuple[str, ...]]] = {
    "statvars": ("dcid",),
    "listed_statvars": ("dcid",),
    "codes": ("dcid",),
    "value_parents": ("dcid",),
    "source_codes": ("provenance", "code"),
}
"""The index blocks the report compares, and the fields keying each block's rows."""

_INDEX_JOINED_FIELDS: Final[frozenset[str]] = frozenset({"provenances"})
"""Row fields the committed index joins from the coverage build, which the report skips."""

_DFO_FINGERPRINT: Final[str] = "dfo_fingerprint"
_KEY_COLUMNS: Final[tuple[str, ...]] = (
    VARIABLE_COLUMN,
    FACET_COLUMN,
    DONOR_COLUMN,
    RECIPIENT_COLUMN,
)
_RELEASE: Final[str] = ":release"
_PUBLISHED: Final[str] = ":published"
_EXAMPLES: Final[int] = 5

_UNREADABLE: Final = (
    OSError,
    ValueError,
    GoogleAPIError,
    CoverageArtifactError,
    CatalogueArtifactError,
)
"""What reading a comparator raises when it cannot be read: a missing or malformed object, a
bucket error, or an artifact its loader refuses."""


def shadow_ready(
    import_stamp: str, *, coverage_uri: str, origin_uri: str
) -> bool | GateUnavailable:
    """Whether the published coverage build's `stamp` and the origin catalogue's
    `dfo_fingerprint` are both `import_stamp`. A pointer or manifest that cannot be read is
    unavailable."""
    try:
        coverage = _coverage_pointer(coverage_uri).stamp
    except _UNREADABLE as exc:
        return GateUnavailable(Dependency.COVERAGE_BUILD, f"coverage pointer unreadable: {exc}")
    try:
        origin = _origin_fingerprint(origin_uri, _origin_pointer(origin_uri))
    except _UNREADABLE as exc:
        return GateUnavailable(Dependency.ORIGIN_CATALOGUE, f"origin identity unreadable: {exc}")
    return coverage == import_stamp and origin == import_stamp


def shadow_diff(
    parts: PlatformParts,
    *,
    import_stamp: str,
    role_axes: Mapping[str, RoleAxis],
    code_index_blocks: bytes,
    coverage_uri: str,
    origin_uri: str,
    committed_index: Traversable | Path | None = None,
) -> GateResult:
    """Diffs the release's `parts` with the comparators built from `import_stamp`.

    `code_index_blocks` is the release's gzipped blocks part, and `committed_index` the index
    it is reported against, the packaged `index.json.gz` by default.
    """
    committed = committed_index if committed_index is not None else default_index_path()
    return merge(
        diff_coverage(parts, import_stamp=import_stamp, role_axes=role_axes, uri=coverage_uri),
        diff_origin_arcs(parts, import_stamp=import_stamp, uri=origin_uri),
        GateResult(reports={"code_index": index_report(code_index_blocks, committed)}),
    )


# ---------------------------------------------------------------- coverage


def diff_coverage(
    parts: PlatformParts, *, import_stamp: str, role_axes: Mapping[str, RoleAxis], uri: str
) -> GateResult:
    """The release's coverage rows and role map against the coverage build published at `uri`,
    row by row on the series key."""
    root = coverage_root(uri)
    try:
        pointer = _coverage_pointer(uri)
        prefix = join_uri(root, VERSIONS_PREFIX, pointer.build_id)
        manifest = _read_manifest(prefix, root=root, pointer=pointer)
        if manifest.stamp != import_stamp:
            return _moved(Dependency.COVERAGE_BUILD, manifest.stamp, import_stamp)
        published = _read_table(prefix, manifest)
        published_roles = _read_role_names(prefix, manifest)
    except _UNREADABLE as exc:
        return GateResult(
            unavailable=(GateUnavailable(Dependency.COVERAGE_BUILD, f"unreadable: {exc}"),)
        )

    try:
        roles = role_map(parts.edges, role_axes)
        rows = coverage_rows(parts.series, roles)
    except PlatformRefusal as refusal:
        return _finding(GateCode.SHADOW_COVERAGE_DIFF, f"the release's coverage refuses: {refusal}")

    findings = []
    if rows.column_names != published.column_names:
        findings.append(
            GateFinding(
                GateCode.SHADOW_COVERAGE_DIFF,
                f"coverage columns {rows.column_names} differ from the published "
                f"{published.column_names}",
            )
        )
    else:
        differ, examples = _row_diff(rows, published)
        if differ:
            findings.append(
                GateFinding(
                    GateCode.SHADOW_COVERAGE_DIFF,
                    f"{differ} coverage row(s) differ from the published build's",
                    examples,
                )
            )
    roles_differ = sorted(
        variable
        for variable in roles.keys() | published_roles.keys()
        if roles.get(variable) != published_roles.get(variable)
    )
    if roles_differ:
        findings.append(
            GateFinding(
                GateCode.SHADOW_COVERAGE_DIFF,
                f"{len(roles_differ)} variable(s) take other role names than the published build's",
                tuple(roles_differ[:_EXAMPLES]),
            )
        )
    return GateResult(
        findings=tuple(findings),
        reports={"shadow_coverage": {"rows": rows.num_rows, "published_rows": published.num_rows}},
    )


def _row_diff(ours: pa.Table, theirs: pa.Table) -> tuple[int, tuple[str, ...]]:
    """How many series keys hold different rows in `ours` and `theirs`, a key held by one
    table only counting as one, and the first few. Keys are compared as positions in the union
    of the two tables' dictionaries, so no key column is decoded."""
    ours = ours.unify_dictionaries().combine_chunks()
    theirs = theirs.unify_dictionaries().combine_chunks()
    dictionaries: dict[str, pa.Array] = {}
    left: dict[str, pa.Array] = {}
    right: dict[str, pa.Array] = {}
    for name in _KEY_COLUMNS:
        mine, other = _dictionary(ours[name]), _dictionary(theirs[name])
        union = pc.unique(pa.concat_arrays([mine.dictionary, other.dictionary.cast(pa.string())]))
        dictionaries[name] = union
        left[name] = pc.take(pc.index_in(mine.dictionary, value_set=union), mine.indices)
        right[name] = pc.take(
            pc.index_in(other.dictionary.cast(pa.string()), value_set=union), other.indices
        )
    values = [name for name in ours.column_names if name not in _KEY_COLUMNS]
    for name in values:
        left[name] = ours[name].combine_chunks()
        right[name] = theirs[name].combine_chunks()
    joined = pa.table(left).join(
        pa.table(right),
        keys=list(_KEY_COLUMNS),
        join_type="full outer",
        left_suffix=_RELEASE,
        right_suffix=_PUBLISHED,
        coalesce_keys=True,
    )
    differs = np.zeros(joined.num_rows, dtype=bool)
    for name in values:
        unequal = pc.not_equal(joined[name + _RELEASE], joined[name + _PUBLISHED])
        differs |= cast("pa.ChunkedArray", unequal).fill_null(pa.scalar(True)).to_numpy()
    found = joined.filter(pa.array(differs))
    head = found.slice(0, _EXAMPLES)
    examples = tuple(
        " ".join(
            str(value)
            for value in (
                dictionaries[name].take(head[name].combine_chunks()).to_pylist()[i]
                for name in _KEY_COLUMNS
            )
        )
        for i in range(head.num_rows)
    )
    return found.num_rows, examples


# ---------------------------------------------------------------- origin arcs


def diff_origin_arcs(parts: PlatformParts, *, import_stamp: str, uri: str) -> GateResult:
    """The release's visible arcs of its DFO variables against the origin catalogue's DFO arc
    table, over every variable either holds with arcs. Targets compare as the mixer names them,
    a literal by its value."""
    try:
        manifest = read_manifest(uri, _origin_pointer(uri).build_id)
        fingerprint = _manifest_fingerprint(manifest.identity_inputs)
        if fingerprint != import_stamp:
            return _moved(Dependency.ORIGIN_CATALOGUE, fingerprint, import_stamp)
        tables = read_arc_tables(uri, manifest)
    except _UNREADABLE as exc:
        return GateResult(
            unavailable=(GateUnavailable(Dependency.ORIGIN_CATALOGUE, f"unreadable: {exc}"),)
        )

    published = {
        node: {(arc.property, arc.dcid if arc.dcid is not None else arc.value) for arc in arcs}
        for node, arcs in tables.dfo.items()
        if arcs
    }
    variables = sorted(published.keys() | set(dfo_variables(parts)))
    ours = release_arcs(visible_edges(parts.edges), parts.nodes, variables)

    pinned = 0
    differ: list[str] = []
    for variable in variables:
        mine = {(prop, name) for prop, names in ours.get(variable, {}).items() for name in names}
        theirs = published.get(variable, set())
        for prop, name in sorted(mine - theirs, key=str):
            differ.append(f"{variable} {prop} {name}: release only")
        for prop, name in sorted(theirs - mine, key=str):
            if prop == PINNED_ARC_PROPERTY and name in IMPORT_ADDED_CONSTRAINT_PROPERTIES:
                pinned += 1
            else:
                differ.append(f"{variable} {prop} {name}: origin catalogue only")
    reports = {"shadow_arcs": {"variables": len(variables), "pinned": pinned}}
    if not differ:
        return GateResult(reports=reports)
    return GateResult(
        findings=(
            GateFinding(
                GateCode.SHADOW_ARC_DIFF,
                f"{len(differ)} arc(s) differ from the origin catalogue's DFO arcs",
                tuple(differ[:_EXAMPLES]),
            ),
        ),
        reports=reports,
    )


# ---------------------------------------------------------------- code index


def index_report(code_index_blocks: bytes, committed: Traversable | Path) -> dict[str, object]:
    """Per block of `INDEX_BLOCK_KEYS`, the rows the release adds, removes and changes against
    the committed index, and which of those name `PINNED_INDEX_VARIABLES`."""
    release = json.loads(gzip.decompress(code_index_blocks))
    index = json.loads(gzip.decompress(committed.read_bytes()))
    report: dict[str, object] = {}
    for block, key in INDEX_BLOCK_KEYS.items():
        old = {_row_key(row, key): row for row in index.get(block, ())}
        new = {_row_key(row, key): row for row in release.get(block, ())}
        added = sorted(new.keys() - old.keys())
        removed = sorted(old.keys() - new.keys())
        fields: Counter[str] = Counter()
        changed = []
        for row_key in sorted(old.keys() & new.keys()):
            moved = {
                name
                for name in old[row_key].keys() | new[row_key].keys()
                if name not in _INDEX_JOINED_FIELDS
                and old[row_key].get(name) != new[row_key].get(name)
            }
            if moved:
                changed.append(row_key)
                fields.update(moved)
        rows = [new[k] for k in added] + [old[k] for k in removed] + [new[k] for k in changed]
        report[block] = {
            "committed": len(old),
            "release": len(new),
            "added": added[:_EXAMPLES],
            "added_count": len(added),
            "removed": removed[:_EXAMPLES],
            "removed_count": len(removed),
            "changed_fields": dict(sorted(fields.items())),
            "pinned": sum(_names_pinned(row) for row in rows),
            "unpinned": sum(not _names_pinned(row) for row in rows),
        }
    return report


def _row_key(row: Mapping[str, object], key: tuple[str, ...]) -> str:
    return " ".join(str(row.get(name)) for name in key)


def _names_pinned(row: object) -> bool:
    text = json.dumps(row, sort_keys=True)
    return any(variable in text for variable in PINNED_INDEX_VARIABLES)


# ---------------------------------------------------------------- comparators


def _coverage_pointer(uri: str) -> CoveragePointer:
    return CoveragePointer.from_json(read_uri_bytes(join_uri(coverage_root(uri), POINTER_FILE)))


def _origin_pointer(uri: str) -> CataloguePointer:
    return CataloguePointer.from_json(read_uri_bytes(join_uri(origin_root(uri), POINTER_FILE)))


def _origin_fingerprint(uri: str, pointer: CataloguePointer) -> str:
    return _manifest_fingerprint(read_manifest(uri, pointer.build_id).identity_inputs)


def _manifest_fingerprint(identity_inputs: Mapping[str, str]) -> str:
    """The import stamp an origin build was built from. Raises `ValueError` for a build that
    records none."""
    fingerprint = identity_inputs.get(_DFO_FINGERPRINT)
    if not fingerprint:
        raise ValueError(f"origin identity records no {_DFO_FINGERPRINT!r}")
    return fingerprint


def _moved(dependency: Dependency, found: str, wanted: str) -> GateResult:
    detail = f"holds import {found!r}, not the release's {wanted!r}"
    return GateResult(unavailable=(GateUnavailable(dependency, detail),))


def _finding(code: GateCode, detail: str) -> GateResult:
    return GateResult(findings=(GateFinding(code, detail),))


def _dictionary(column: pa.ChunkedArray) -> pa.DictionaryArray:
    return cast("pa.DictionaryArray", column.combine_chunks())


__all__ = [
    "INDEX_BLOCK_KEYS",
    "PINNED_ARC_PROPERTY",
    "PINNED_INDEX_VARIABLES",
    "diff_coverage",
    "diff_origin_arcs",
    "index_report",
    "shadow_diff",
    "shadow_ready",
]
