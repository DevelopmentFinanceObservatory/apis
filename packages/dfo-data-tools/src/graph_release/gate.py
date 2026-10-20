"""The gate a release passes before the pointer moves.

`run_gate` reads the staged platform files back and checks what ships: the invariants, the DFO
parts rebuilt byte for byte from the files, the serving views built from the files, the mixer
differential and the SDMX counts. The shadow diff (`graph_release.shadow`) returns a
`GateResult` of its own, which `merge` joins.

A result has two kinds of failure. A finding is deterministic: the same inputs fail the same
way, so the builder records them as the last failed build and does not retry them. An
unavailable entry means a dependency could not answer, so the builder records nothing and
retries on the next tick. An unavailable dependency never counts as a pass. Reports carry
counts that inform and decide nothing.

The samples are seeded, so one seed and one release always check the same variables and rows.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from dc_graph import parse_arcs
from dc_graph.errors import GraphCapExceededError, GraphReadError
from dc_graph.visible_arcs import mixer_target, visible_mask
from dc_sdmx.errors import SdmxClientError, SdmxQueryError
from dc_sdmx.results import SdmxDataResult
from dfo_platform.roles import RoleAxis
from graph_release.coverage import coverage_rows, role_map
from graph_release.declarations import Declarations
from graph_release.dfo import build_dfo_parts, dfo_inputs
from graph_release.format import DFO, PLATFORM, LoadedRelease, StagedPart, join_part_files
from graph_release.platform import PlatformParts, PlatformRefusal
from graph_release.views import (
    CODE_INDEX_PART,
    DECLARATIONS_PART,
    ViewRefused,
    code_indexes,
    coverage_index,
    declared_role_axes,
    read_document,
)
from series_coverage.schema import VARIABLE_COLUMN

MIXER_CHANGED_CAP: Final[int] = 500
"""The most variables whose arcs changed since the previous release the mixer differential
reads."""

MIXER_RANDOM: Final[int] = 100
"""Variables the mixer differential reads beyond the changed ones, drawn from the rest."""

MIXER_ONLY_VARIABLES: Final[int] = 25
"""Sampled variables whose `->*` arcs the `mixer_only` report reads."""

SDMX_ROWS: Final[int] = 30
"""`series` rows whose `observation_count` the SDMX check compares with `/data`."""

_OBSERVATION_PROPERTIES: Final[str] = "observationProperties"
_OBSERVATION_ABOUT: Final[str] = "observationAbout"
_STATISTICAL_VARIABLE: Final[str] = "StatisticalVariable"
_ENTITY_COLUMNS: Final[tuple[str, ...]] = ("entity1", "entity2", "entity3")
_BASE_PARTIES: Final[str] = "base_parties"
_EXAMPLES: Final[int] = 5


class GateCode(StrEnum):
    """What a finding names. The platform build's refusals (`PlatformRefusal.code`) use the same
    values."""

    POPCOUNT_MISMATCH = "popcount_mismatch"
    UNKNOWN_FACET = "unknown_facet"
    SUBSET_CHAIN = "subset_chain"
    MISSING_ROLE_DECLARATION = "missing_role_declaration"
    DANGLING_EDGE = "dangling_edge"
    AUTHORED_MISMATCH = "authored_mismatch"
    DFO_REBUILD_MISMATCH = "dfo_rebuild_mismatch"
    MIXER_ARC_MISMATCH = "mixer_arc_mismatch"
    SDMX_COUNT_MISMATCH = "sdmx_count_mismatch"
    SHADOW_COVERAGE_DIFF = "shadow_coverage_diff"
    SHADOW_ARC_DIFF = "shadow_arc_diff"
    UNSUPPORTED_OBSERVATION_PERIOD = "unsupported_observation_period"
    INCONSISTENT_FACET = "inconsistent_facet"
    CODE_INDEX_EXPORT_REFUSED = "code_index_export_refused"
    COVERAGE_VIEW_REFUSED = "coverage_view_refused"
    CODE_INDEX_UNLOADABLE = "code_index_unloadable"


class Dependency(StrEnum):
    """What an unavailable entry could not read."""

    MIXER = "mixer"
    SDMX = "sdmx"
    BUCKET = "bucket"
    ADMIN_API = "admin_api"
    COVERAGE_BUILD = "coverage_build"
    ORIGIN_CATALOGUE = "origin_catalogue"
    SPANNER = "spanner"


@dataclass(frozen=True)
class GateFinding:
    """A deterministic failure. `examples` holds a few of the rows or ids at fault, sorted."""

    code: GateCode
    detail: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateUnavailable:
    """A dependency that could not answer."""

    dependency: Dependency
    detail: str


@dataclass(frozen=True)
class GateResult:
    """One gate run. A finding decides the outcome when there is one, since it is deterministic.
    Otherwise any unavailable entry does."""

    findings: tuple[GateFinding, ...] = ()
    unavailable: tuple[GateUnavailable, ...] = ()
    reports: Mapping[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.findings and not self.unavailable

    def summary(self) -> dict[str, object]:
        """The result as JSON values, for the manifest's `gates` and the job's events."""
        return {
            "findings": [
                {"code": str(f.code), "detail": f.detail, "examples": list(f.examples)}
                for f in self.findings
            ],
            "unavailable": [
                {"dependency": str(u.dependency), "detail": u.detail} for u in self.unavailable
            ],
            "reports": dict(self.reports),
        }


def merge(*results: GateResult) -> GateResult:
    """The findings, unavailable entries and reports of `results` in order. Report keys must be
    distinct."""
    reports: dict[str, object] = {}
    for result in results:
        if clash := reports.keys() & result.reports.keys():
            raise ValueError(f"gate results report {sorted(clash)} twice")
        reports.update(result.reports)
    return GateResult(
        findings=tuple(f for result in results for f in result.findings),
        unavailable=tuple(u for result in results for u in result.unavailable),
        reports=reports,
    )


def refused(refusal: PlatformRefusal) -> GateResult:
    """The finding for a platform build that refused its snapshot."""
    return GateResult(findings=(GateFinding(GateCode(refusal.code), str(refusal)),))


class ArcReader(Protocol):
    """`dc_graph.GraphClient`'s arc reads. `node_arcs_many` answers each dcid's values per
    property, `None` for a node the graph does not hold; `node_arcs_batch` answers each dcid's
    `->*` arcs in the `/v2/node` wire form, `{property: {"nodes": [{"dcid" or "value": ...}]}}`,
    `None` for a node with none."""

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> Mapping[str, Mapping[str, tuple[str, ...]] | None]: ...

    def node_arcs_batch(self, dcids: list[str]) -> Mapping[str, Mapping[str, object] | None]: ...


class SeriesReader(Protocol):
    """`dc_sdmx.client.SdmxClient.data`."""

    def data(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, Sequence[str]],
        facet_ids: Sequence[str] | None = None,
    ) -> SdmxDataResult: ...


def run_gate(
    staged: Sequence[StagedPart],
    *,
    declarations: Declarations,
    previous: LoadedRelease | None,
    graph: ArcReader,
    sdmx: SeriesReader,
    seed: str,
) -> GateResult:
    """Gates the release whose files are `staged`.

    The platform parts are read back from the staged files, so every check covers the bytes
    that would be uploaded. `previous` is the serving release, `None` when none is published.
    The mixer differential reads the variables whose arcs changed since it. `seed` (the import
    stamp) fixes the samples.
    """
    parts = read_back(staged)
    return merge(
        GateResult(
            findings=check_invariants(parts, role_axes=declarations.role_axes),
            reports={"rows": _row_counts(parts)},
        ),
        GateResult(findings=check_dfo_rebuild(parts, staged=staged, declarations=declarations)),
        GateResult(findings=check_views(parts, staged=staged)),
        check_mixer(
            parts,
            previous_edges=None if previous is None else previous.tables["edges"],
            graph=graph,
            seed=seed,
        ),
        check_sdmx(parts, sdmx=sdmx, seed=seed),
    )


def read_back(staged: Sequence[StagedPart]) -> PlatformParts:
    """The platform parts as the staged files hold them, each part's files mapped, joined and
    put back in key order without decoding a dictionary (`join_part_files`)."""
    files: dict[str, list[pa.Table]] = {}
    for part in staged:
        if part.ref.layer == PLATFORM:
            table = pa.ipc.open_file(pa.memory_map(str(part.path), "r")).read_all()
            files.setdefault(part.ref.name, []).append(table)
    return PlatformParts(
        **{
            name: join_part_files(files[name], name=name)
            for name in ("series", "facets", "edges", "nodes", "imports")
        }
    )


# ---------------------------------------------------------------- invariants


def check_invariants(
    parts: PlatformParts, *, role_axes: Mapping[str, RoleAxis]
) -> tuple[GateFinding, ...]:
    """The invariants over the platform parts: mask popcounts within the observation counts,
    facet ids, coverage variables
    within the variables with arcs, a role declaration in `role_axes` for every coverage
    variable, and no dangling edge.

    The rest of the subset chain, variables with arcs within the live variables, needs the
    live variables, which come from `TimeSeries` and which no platform part holds. `series`
    lacks a live variable with no observation, and `edges` holds the arcs of the second-hop
    variables the closure keeps."""
    findings: list[GateFinding] = []
    series = parts.series

    counted = sum(
        np.bitwise_count(_words(series[name])).astype(np.int64)
        for name in series.column_names
        if name.startswith("period_mask_")
    )
    # A mask sets one bit per year inside it, so two dates in one year, or a year outside the
    # mask, count observations it does not.
    wrong = np.flatnonzero(series["observation_count"].to_numpy().astype(np.int64) < counted)
    if len(wrong):
        rows = series.take(wrong[:_EXAMPLES])
        findings.append(
            GateFinding(
                GateCode.POPCOUNT_MISMATCH,
                f"{len(wrong)} series row(s) whose period masks count more years than observations",
                tuple(
                    f"{row['variable_measured']} {row['entity1']} {row['facet_id']}"
                    for row in rows.to_pylist()
                ),
            )
        )

    unknown = _missing(_unique(series["facet_id"]), parts.facets["facet_id"])
    if unknown:
        findings.append(
            GateFinding(
                GateCode.UNKNOWN_FACET,
                f"{len(unknown)} facet id(s) in series have no facets row",
                _first(unknown),
            )
        )

    try:
        coverage = coverage_rows(series, role_map(parts.edges, role_axes))
    except PlatformRefusal as refusal:
        findings.append(GateFinding(GateCode(refusal.code), str(refusal)))
    else:
        lacking = _missing(_unique(coverage[VARIABLE_COLUMN]), _variables_with_arcs(parts))
        if lacking:
            findings.append(
                GateFinding(
                    GateCode.SUBSET_CHAIN,
                    f"{len(lacking)} coverage variable(s) have no arcs as a variable",
                    _first(lacking),
                )
            )

    ends = pa.chunked_array(
        [_decoded(parts.edges["subject_id"]), _decoded(parts.edges["object_id"])]
    )
    if dangling := _missing(_unique(ends), parts.nodes["subject_id"]):
        findings.append(
            GateFinding(
                GateCode.DANGLING_EDGE,
                f"{len(dangling)} edge end(s) have no nodes row",
                _first(dangling),
            )
        )
    return tuple(findings)


def _observation_properties(edges: pa.Table) -> dict[str, tuple[str, ...]]:
    """Each variable's `observationProperties` values, sorted into the order of its series'
    entities."""
    rows = edges.filter(pc.equal(_decoded(edges["predicate"]), pa.scalar(_OBSERVATION_PROPERTIES)))
    found: dict[str, set[str]] = {}
    for subject, role in zip(
        cast("list[str]", _decoded(rows["subject_id"]).to_pylist()),
        cast("list[str]", _decoded(rows["object_id"]).to_pylist()),
        strict=True,
    ):
        found.setdefault(subject, set()).add(role)
    return {subject: tuple(sorted(roles)) for subject, roles in found.items()}


def _variables_with_arcs(parts: PlatformParts) -> pa.Array:
    """Edge subjects whose `nodes` row is typed `StatisticalVariable`."""
    types = cast("pa.ListArray", parts.nodes["types"].combine_chunks())
    typed = pc.equal(_decoded_array(types.flatten()), pa.scalar(_STATISTICAL_VARIABLE))
    variables = parts.nodes["subject_id"].take(
        pc.unique(pc.list_parent_indices(types).filter(typed))
    )
    subjects = _unique(parts.edges["subject_id"])
    return subjects.filter(pc.is_in(subjects, value_set=variables))


# ---------------------------------------------------------------- DFO rebuild


def check_dfo_rebuild(
    parts: PlatformParts, *, staged: Sequence[StagedPart], declarations: Declarations
) -> tuple[GateFinding, ...]:
    """Rebuilds the DFO parts from `parts`, `declarations` and the staged `base_parties` file,
    the mixer's answer the build stored, and compares the bytes with the staged DFO files."""
    written = {part.ref.name: part.path.read_bytes() for part in staged if part.ref.layer == DFO}
    base_parties = written.get(_BASE_PARTIES)
    if base_parties is None:
        return (GateFinding(GateCode.DFO_REBUILD_MISMATCH, "no base_parties part is staged"),)
    try:
        rebuilt = build_dfo_parts(dfo_inputs(parts, declarations), base_parties).blobs()
    except PlatformRefusal as refusal:
        return (
            GateFinding(
                GateCode.DFO_REBUILD_MISMATCH,
                f"the DFO parts do not rebuild from the staged platform files: {refusal}",
                (refusal.code,),
            ),
        )
    differ = sorted(
        name for name in rebuilt.keys() | written.keys() if rebuilt.get(name) != written.get(name)
    )
    if not differ:
        return ()
    return (
        GateFinding(
            GateCode.DFO_REBUILD_MISMATCH,
            f"{len(differ)} DFO part(s) rebuild from the staged platform files to other bytes",
            tuple(differ),
        ),
    )


# ---------------------------------------------------------------- serving views


def check_views(parts: PlatformParts, *, staged: Sequence[StagedPart]) -> tuple[GateFinding, ...]:
    """Builds the serving views from `parts` and the staged DFO files through
    `graph_release.views`, as a serving process builds them from the published release, and
    names each view that does not build: `coverage_view_refused` for the `CoverageIndex`,
    `code_index_unloadable` for the code and measure indexes.

    The coverage view takes its role axes from the staged `declarations` file. The build year
    only bounds what a read elects, so the view is built without one."""
    written = {part.ref.name: part.path for part in staged if part.ref.layer == DFO}
    findings: list[GateFinding] = []
    try:
        declarations = written.get(DECLARATIONS_PART)
        if declarations is None:
            raise ViewRefused(f"no {DECLARATIONS_PART} part is staged")
        coverage_index(
            series=parts.series,
            edges=parts.edges,
            facets=parts.facets,
            role_axes=declared_role_axes(declarations.read_bytes()),
            build_year=None,
        )
    except ViewRefused as refusal:
        findings.append(GateFinding(GateCode.COVERAGE_VIEW_REFUSED, str(refusal)))
    try:
        index = written.get(CODE_INDEX_PART)
        if index is None:
            raise ViewRefused(f"no {CODE_INDEX_PART} part is staged")
        code_indexes(read_document(index.read_bytes()))
    except ViewRefused as refusal:
        findings.append(GateFinding(GateCode.CODE_INDEX_UNLOADABLE, str(refusal)))
    return tuple(findings)


# ---------------------------------------------------------------- mixer


def check_mixer(
    parts: PlatformParts, *, previous_edges: pa.Table | None, graph: ArcReader, seed: str
) -> GateResult:
    """Compares the release's visible arcs with the mixer's for up to `MIXER_CHANGED_CAP`
    variables whose arcs changed since the previous release's `previous_edges`, every variable
    counting as changed when it is `None`, plus `MIXER_RANDOM` others.

    The comparison covers the properties the sampled release arcs hold. Variables with an
    authored `nodes` row are left out: the mixer answers them from the base graph, which the
    release does not copy. The variables the mixer omits are read once more, and a second
    omission is a mismatch. A read error is unavailable.

    The `mixer_only` report reads the `->*` arcs of `MIXER_ONLY_VARIABLES` of the sampled
    variables, chosen by `seed`, and counts, per property outside the compared ones, those with
    a DFO-visible target. It decides nothing, since `->*` also answers arcs from the base graph
    the release does not copy; a failed read is reported as `unread`.
    """
    edges = visible_edges(parts.edges)
    candidates = dfo_variables(parts)
    if previous_edges is None:
        changed = list(candidates)
    else:
        moved = _changed_subjects(edges, visible_edges(previous_edges))
        changed = [v for v in candidates if v in moved]
    changed = _seeded(changed, seed)[:MIXER_CHANGED_CAP]
    rest = _seeded(sorted(set(candidates) - set(changed)), seed)[:MIXER_RANDOM]
    sample = sorted([*changed, *rest])
    reports: dict[str, object] = {"mixer": {"changed": len(changed), "sampled": len(sample)}}
    if not sample:
        return GateResult(reports=reports)

    release = release_arcs(edges, parts.nodes, sample)
    properties = sorted({prop for arcs in release.values() for prop in arcs})
    try:
        served = dict(graph.node_arcs_many(sample, properties=properties))
        if omitted := [variable for variable in sample if served.get(variable) is None]:
            served.update(graph.node_arcs_many(omitted, properties=properties))
    except (GraphReadError, GraphCapExceededError) as exc:
        return GateResult(
            unavailable=(GateUnavailable(Dependency.MIXER, f"node_arcs_many failed: {exc}"),),
            reports=reports,
        )
    reports["mixer_only"] = _mixer_only(
        graph, sorted(_seeded(sample, seed)[:MIXER_ONLY_VARIABLES]), compared=set(properties)
    )

    mismatched = []
    for variable in sample:
        ours = release.get(variable, {})
        theirs = served.get(variable)
        if theirs is None:
            mismatched.append(f"{variable}: the mixer holds no node")
            continue
        for prop in properties:
            have, want = ours.get(prop, set()), set(theirs.get(prop, ()))
            if have != want:
                mismatched.append(
                    f"{variable} {prop}: release only {sorted(have - want)}, "
                    f"mixer only {sorted(want - have)}"
                )
    if not mismatched:
        return GateResult(reports=reports)
    return GateResult(
        findings=(
            GateFinding(
                GateCode.MIXER_ARC_MISMATCH,
                f"variable properties differing from the mixer's arcs: {len(mismatched)}",
                tuple(mismatched[:_EXAMPLES]),
            ),
        ),
        reports=reports,
    )


def _mixer_only(graph: ArcReader, variables: list[str], *, compared: set[str]) -> dict[str, object]:
    """How many `variables` the report read, and per property outside `compared`, how many of
    them the mixer's `->*` arcs give a DFO-visible target for."""
    try:
        served = graph.node_arcs_batch(variables)
    except (GraphReadError, GraphCapExceededError) as exc:
        return {"variables": len(variables), "unread": str(exc)}
    found: Counter[str] = Counter()
    for arcs in served.values():
        if not arcs:
            continue
        # `parse_arcs` keeps only the targets `dfo_visible` keeps.
        extra = parse_arcs(dict(arcs), [prop for prop in arcs if prop not in compared])
        found.update(prop for prop, values in extra.items() if values)
    return {"variables": len(variables), "properties": dict(sorted(found.items()))}


def visible_edges(edges: pa.Table) -> pa.Table:
    """`edges` decoded to plain strings, less what either visibility stage hides."""
    decoded = pa.table(
        {name: _decoded(edges[name]) for name in ("subject_id", "predicate", "object_id")}
    )
    return decoded.filter(
        visible_mask(decoded["predicate"].combine_chunks(), decoded["object_id"].combine_chunks())
    )


def _changed_subjects(edges: pa.Table, before: pa.Table) -> set[str]:
    """Subjects with an arc in one table and not the other."""
    keys = ["subject_id", "predicate", "object_id"]
    added = edges.join(before, keys=keys, join_type="left anti")["subject_id"]
    removed = before.join(edges, keys=keys, join_type="left anti")["subject_id"]
    return set(cast("list[str]", added.to_pylist())) | set(cast("list[str]", removed.to_pylist()))


def release_arcs(
    edges: pa.Table, nodes: pa.Table, variables: Sequence[str]
) -> dict[str, dict[str, set[str]]]:
    """Each variable's arcs as the mixer names them: a literal target by its value."""
    rows = edges.filter(pc.is_in(edges["subject_id"], value_set=pa.array(variables)))
    targets = nodes.filter(pc.is_in(nodes["subject_id"], value_set=rows["object_id"]))
    described = {
        row["subject_id"]: row
        for row in targets.select(["subject_id", "value", "types"]).to_pylist()
    }
    arcs: dict[str, dict[str, set[str]]] = {}
    for row in rows.to_pylist():
        target = described.get(row["object_id"])
        name = (
            row["object_id"]
            if target is None
            else mixer_target(row["object_id"], value=target["value"], types=target["types"])
        )
        arcs.setdefault(row["subject_id"], {}).setdefault(row["predicate"], set()).add(name)
    return arcs


def dfo_variables(parts: PlatformParts) -> list[str]:
    """The live variables, less those with an authored `nodes` row, sorted. These are the variables whose
    arcs the release copies from DFO's Spanner."""
    nodes = parts.nodes
    return _missing(
        _unique(parts.series["variable_measured"]), nodes["subject_id"].filter(nodes["authored"])
    )


# ---------------------------------------------------------------- SDMX


def check_sdmx(parts: PlatformParts, *, sdmx: SeriesReader, seed: str) -> GateResult:
    """Compares `observation_count` with the rows SDMX `/data` returns for `SDMX_ROWS` seeded
    `series` rows. Each row's entities fill its variable's sorted `observationProperties`, or
    `observationAbout` for a variable with none. A refused query is a finding; any other SDMX
    error is unavailable."""
    series = parts.series
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8]))
    picked = np.sort(rng.choice(series.num_rows, min(SDMX_ROWS, series.num_rows), replace=False))
    rows = series.take(picked).to_pylist()
    roles_of = _observation_properties(parts.edges)
    reports = {"sdmx": {"sampled": len(rows)}}

    mismatched = []
    for row in rows:
        variable = row["variable_measured"]
        roles = roles_of.get(variable) or (_OBSERVATION_ABOUT,)
        label = f"{variable} {row['entity1']} {row['extra_entities_id']} {row['facet_id']}"
        entities = [row.get(column) for column in _ENTITY_COLUMNS[: len(roles)]]
        if len(roles) > len(_ENTITY_COLUMNS) or None in entities:
            mismatched.append(f"{label}: roles {list(roles)} and entities {entities} disagree")
            continue
        try:
            result = sdmx.data(
                variables=[variable],
                entities={role: [cast("str", e)] for role, e in zip(roles, entities, strict=True)},
                facet_ids=[row["facet_id"]],
            )
        except SdmxQueryError as exc:
            mismatched.append(f"{label}: query refused: {exc}")
            continue
        except SdmxClientError as exc:
            return GateResult(
                unavailable=(GateUnavailable(Dependency.SDMX, f"/data failed: {exc}"),),
                reports=reports,
            )
        if len(result.rows) != row["observation_count"]:
            mismatched.append(
                f"{label}: series counts {row['observation_count']}, /data returns "
                f"{len(result.rows)}"
            )
    if not mismatched:
        return GateResult(reports=reports)
    return GateResult(
        findings=(
            GateFinding(
                GateCode.SDMX_COUNT_MISMATCH,
                f"{len(mismatched)} of {len(rows)} sampled series row(s) disagree with /data",
                tuple(mismatched[:_EXAMPLES]),
            ),
        ),
        reports=reports,
    )


# ---------------------------------------------------------------- helpers


def _words(column: pa.ChunkedArray) -> np.ndarray:
    """A period mask column's words as unsigned integers, a null word as no bits."""
    return column.fill_null(pa.scalar(0, pa.int64())).to_numpy().view(np.uint64)


def _seeded(values: Sequence[str], seed: str) -> list[str]:
    """`values` in an order fixed by `seed`."""
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest())


def _decoded(column: pa.ChunkedArray) -> pa.Array:
    return _decoded_array(column.combine_chunks())


def _decoded_array(array: pa.Array) -> pa.Array:
    if pa.types.is_dictionary(array.type):
        return cast("pa.DictionaryArray", array).dictionary_decode()
    return array


def _unique(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    if isinstance(column, pa.ChunkedArray):
        column = _decoded(column)
    return pc.unique(_decoded_array(column)).drop_null()


def _missing(values: pa.Array, known: pa.ChunkedArray | pa.Array) -> list[str]:
    """`values` absent from `known`, sorted."""
    known = _decoded(known) if isinstance(known, pa.ChunkedArray) else _decoded_array(known)
    absent = values.filter(pc.invert(pc.is_in(values, value_set=known)))
    return sorted(cast("list[str]", absent.to_pylist()))


def _first(values: list[str]) -> tuple[str, ...]:
    return tuple(values[:_EXAMPLES])


def _row_counts(parts: PlatformParts) -> dict[str, int]:
    return {name: table.num_rows for name, table in parts.tables().items()}


__all__ = [
    "MIXER_CHANGED_CAP",
    "MIXER_ONLY_VARIABLES",
    "MIXER_RANDOM",
    "SDMX_ROWS",
    "ArcReader",
    "Dependency",
    "GateCode",
    "GateFinding",
    "GateResult",
    "GateUnavailable",
    "SeriesReader",
    "check_dfo_rebuild",
    "check_invariants",
    "check_mixer",
    "check_sdmx",
    "check_views",
    "dfo_variables",
    "merge",
    "read_back",
    "refused",
    "release_arcs",
    "run_gate",
    "visible_edges",
]
