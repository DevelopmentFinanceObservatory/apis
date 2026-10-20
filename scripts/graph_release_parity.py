"""Diff what a graph release serves against what the old paths serve at the same import.

Two gates, one per flag, each run in an environment before its flag is switched on there. Each
opens the release the pointer under `--release-uri` names.

`coverage`, the gate before `GRAPH_RELEASE_COVERAGE`, builds the release's serving views with
`graph_release.views`, the functions a serving process opens it with, and compares them with
the published coverage build and a fresh code index export of the same import:

- the release's `CoverageIndex` against the published coverage build under `--coverage-uri`:
  every row (`graph_release.shadow.row_diff`), each variable's role names and provenances, each
  facet's metadata, and the horizon and build year;
- the release's `code_index_blocks` document against a fresh `export_index` (the function
  `scripts/export_code_index.py` runs) over the live mixer at `--graph-base`, joined with that
  same coverage build, every block row by row.

`origin`, the gate before `GRAPH_RELEASE_ORIGIN`, builds the release's origin view
(`graph_release.origin`), loads the origin catalogue build under `--origin-catalogue-uri` as a
serving process does with the flag on, and reads each answer the flag moves twice, flag on with
the release bound and flag off:

- `held_family` on every family key the build or the release holds: members in order, values,
  shared values and the total (`origin.families`);
- `held_arcs` of every DFO variable, each row with its target's name and types
  (`origin.arcs`), and each target those arcs name, with whether `arc_target` and `arc_label`
  answer it (`origin.arc_targets`);
- every member walk of every DFO family the flag on walks over the release's graph reader,
  against the same walk over the mixer with the catalogue bound, as the flag off walks it:
  members and family in order, refusals included (`origin.members`);
- the skills index built over the mixer: its roster, each skill's attachments and its reach
  (`skills.roster`, `skills.attachments`, `skills.reach`).

The report also counts the families the release-held rule cannot see
(`ReleaseOrigin.holds_family`), which no category counts unless they change a walk: base Data
Commons variables of a release-held five-tuple that the catalogue does not list, found by the
mixer's reverse read, and variables the release's edges type `StatisticalVariable` with no
observation in its series.

A gate refuses (exit 2) before comparing anything when the build it compares with and the
release come from different ingestion workflow executions (`import_mismatch`, or
`coverage_ingestion_unrecorded` and `catalogue_ingestion_unrecorded` for a build that records
none), or when the DC service serves another revision than the release read, before or after
the mixer reads (`serving_revision_moved`), since the mixer would then answer from another
import. `coverage` also refuses when this checkout's declarations hash differs from the
release's (`declarations_moved`), since a fresh export would then be built from other
declarations, and when the export refuses (`export_refused`).

The JSON report gives, per category, how many items were compared, how many differ and the
first `--examples` differences. The exit status is 0 when nothing differs, 1 when anything does.

Reads only: the release, coverage and origin catalogue buckets, the mixer, the UN deployment
for UN dcids, and the Cloud Run Admin API's `run.services.get` on the DC service.

Usage:
    uv run python scripts/graph_release_parity.py coverage \\
        --release-uri gs://dfo-dev-graph-release-dev-fin-observatory/ \\
        --coverage-uri gs://dfo-dev-coverage-dev-fin-observatory/ \\
        --graph-base https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app \\
        --datacommons-service projects/dev-fin-observatory/locations/northamerica-northeast1/services/dfo-dev-dc-datacommons-service \\
        --out parity-dev.json
    uv run python scripts/graph_release_parity.py origin \\
        --release-uri gs://dfo-dev-graph-release-dev-fin-observatory/ \\
        --origin-catalogue-uri gs://dfo-dev-origin-catalogue-dev-fin-observatory/ \\
        --graph-base https://dfo-dev-dc-datacommons-service-zms5lj4zhq-nn.a.run.app \\
        --datacommons-service projects/dev-fin-observatory/locations/northamerica-northeast1/services/dfo-dev-dc-datacommons-service \\
        --out parity-origin-dev.json
"""

from __future__ import annotations

import argparse
import contextvars
import json
import os
import sys
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.compute as pc

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from artifact_publication.uri import join_uri  # noqa: E402
from code_index.declarations import DATASETS, LISTED_DATASETS  # noqa: E402
from code_index.export import ExportRefused, content_hash  # noqa: E402
from dc_graph import GraphReadError  # noqa: E402
from graph_release.context import bound_release  # noqa: E402
from graph_release.coverage import coverage_rows, role_map  # noqa: E402
from graph_release.declarations import DECLARATIONS  # noqa: E402
from graph_release.format import LoadedRelease, Manifest, open_release  # noqa: E402
from graph_release.job import builder_code_hash  # noqa: E402
from graph_release.origin import ReleaseOrigin, release_origin  # noqa: E402
from graph_release.service import Release  # noqa: E402
from graph_release.serving_revision import CloudRunAdmin  # noqa: E402
from graph_release.shadow import INDEX_BLOCK_KEYS, row_diff  # noqa: E402
from graph_release.views import (  # noqa: E402
    CODE_INDEX_PART,
    DECLARATIONS_PART,
    ViewRefused,
    declared_role_axes,
    read_document,
    release_code_indexes,
    release_coverage_index,
)
from origin_catalogue.arc_reads import DFO_VARIABLE_TYPE  # noqa: E402
from origin_catalogue.catalogue import CatalogueSnapshot  # noqa: E402
from origin_catalogue.context import bound_catalogue  # noqa: E402
from origin_catalogue.families import FamilySummary  # noqa: E402
from origin_catalogue.schema import CatalogueManifest  # noqa: E402
from origin_catalogue.un_mirror import UnArc  # noqa: E402
from scripts.export_code_index import coverage_join, export_index, export_reader  # noqa: E402
from series_coverage.index import CoverageIndex  # noqa: E402
from series_coverage.loader import (  # noqa: E402
    LoadedCoverage,
    format_root,
    load_coverage,
    read_published_pointer,
    read_table,
)
from series_coverage.schema import VERSIONS_PREFIX, CoverageManifest  # noqa: E402
from skills_index.types import SkillsIndex  # noqa: E402
from statvar_members import (  # noqa: E402
    DEFAULT_SUBJECT_CEILING,
    AbsentReading,
    DegenerateObservationProperties,
    FiveTuple,
    MemberArcReader,
    MemberEnumerationTooLarge,
    MemberQueryUnanchored,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    SlotBinding,
    enumerate_members,
    five_tuple_arcs_match,
    five_tuple_reverse_filters,
)

_ROW_KEYS: Final[Mapping[str, tuple[str, ...]]] = {**INDEX_BLOCK_KEYS, "parties": ("dcid",)}
"""The fields keying a code index block's rows. A list block not named here keys each row by
its canonical JSON, so a changed row reads as one row on each side."""

_RELEASE_PARTS: Final[frozenset[str]] = frozenset(
    {"series", "edges", "facets", DECLARATIONS_PART, CODE_INDEX_PART}
)
_DEFAULT_EXAMPLES: Final[int] = 10

ORIGIN_FLAG: Final[str] = "GRAPH_RELEASE_ORIGIN"
COVERAGE_FLAG: Final[str] = "GRAPH_RELEASE_COVERAGE"
_ORIGIN_PARTS: Final[frozenset[str]] = frozenset({"series", "edges", "nodes"})
_FIVE_TUPLE_ARCS: Final[tuple[str, ...]] = (
    "populationType",
    "measuredProperty",
    "statType",
    "measurementQualifier",
    "measurementDenominator",
)
_OBSERVATION_PROPERTIES: Final[str] = "observationProperties"
_UNBOUND: Final[SlotBinding] = ("unbound",)
_WALK_REFUSALS: Final = (
    MemberQueryUnanchored,
    MemberEnumerationTooLarge,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    DegenerateObservationProperties,
    GraphReadError,
)
"""What a member walk raises for a five-tuple: the walk's refusals, and a graph read that
failed past its retries, which compares as one."""


class Refused(Exception):
    """A run that cannot compare the release with the old path; `code` names why."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass
class Category:
    """The differences found in one category, the first `limit` kept."""

    limit: int
    compared: int = 0
    differences: int = 0
    examples: list[object] = field(default_factory=list)

    def add(self, difference: object) -> None:
        self.differences += 1
        if len(self.examples) < self.limit:
            self.examples.append(difference)

    def to_json(self) -> dict[str, object]:
        return {
            "compared": self.compared,
            "differences": self.differences,
            "examples": self.examples,
        }


# ---------------------------------------------------------------- preconditions


def check_same_import(
    release: Manifest,
    build: CoverageManifest | CatalogueManifest,
    *,
    artifact: str = "coverage",
) -> None:
    """Raises `Refused` unless `build`, a coverage or origin catalogue build named by
    `artifact`, records the release's ingestion execution."""
    if build.ingestion is None:
        raise Refused(
            f"{artifact}_ingestion_unrecorded",
            f"{artifact} build {build.build_id} records no ingestion execution",
        )
    if build.ingestion.workflow_execution_id != release.import_execution_id:
        raise Refused(
            "import_mismatch",
            f"{artifact} build {build.build_id} was built from execution "
            f"{build.ingestion.workflow_execution_id}, release {release.release_id} from "
            f"{release.import_execution_id}",
        )


def check_declarations(release: Manifest) -> None:
    """Raises `Refused` unless this checkout's declarations hash is the release's."""
    found = DECLARATIONS.declarations_hash()
    if found != release.declarations_hash:
        raise Refused(
            "declarations_moved",
            f"this checkout's declarations hash {found} is not the release's "
            f"{release.declarations_hash}",
        )


def check_serving_revision(release: Manifest, serving: Callable[[], str]) -> None:
    """Raises `Refused` unless `serving()` names the revision the release read."""
    token = serving()
    if token != release.dc_serving_revision:
        raise Refused(
            "serving_revision_moved",
            f"the DC service serves {token}, the release read {release.dc_serving_revision}",
        )


# ---------------------------------------------------------------- coverage


def coverage_parity(
    release: LoadedRelease,
    view: CoverageIndex,
    published: LoadedCoverage,
    published_rows: pa.Table,
    *,
    examples: int,
) -> dict[str, Category]:
    """The release's coverage view `view` against the published build: its rows, role names,
    variable provenances, facet metadata, horizon and build year. `published_rows` is the
    build's table as `series_coverage.loader.read_table` reads it."""
    roles = role_map(release.tables["edges"], declared_role_axes(release.blobs[DECLARATIONS_PART]))
    rows = Category(examples, compared=max(view.num_rows, published.index.num_rows))
    count, keys = row_diff(
        coverage_rows(release.tables["series"], roles), published_rows, examples=examples
    )
    rows.differences = count
    rows.examples = list(keys)

    theirs = published.index
    found = {"coverage.rows": rows}
    for name, ours_map, theirs_map in (
        ("coverage.role_names", view.role_names, theirs.role_names),
        (
            "coverage.variable_provenances",
            view.variable_provenances(),
            theirs.variable_provenances(),
        ),
        ("coverage.facet_metadata", view.facet_metadata, theirs.facet_metadata),
    ):
        found[name] = _mapping_diff(ours_map, theirs_map, examples=examples)

    index = Category(examples, compared=2)
    for name, ours_value, theirs_value in (
        ("horizon", asdict(view.horizon), asdict(theirs.horizon)),
        ("build_year", view.build_year, theirs.build_year),
    ):
        if ours_value != theirs_value:
            index.add({"field": name, "release": ours_value, "published": theirs_value})
    found["coverage.index"] = index
    return found


def _mapping_diff(
    ours: Mapping[str, object],
    theirs: Mapping[str, object],
    *,
    examples: int,
    other: str = "published",
) -> Category:
    """Each key of `ours` or `theirs` whose values differ, the release's value beside the one
    `theirs` holds under the label `other`."""
    keys = sorted(ours.keys() | theirs.keys())
    category = Category(examples, compared=len(keys))
    for key in keys:
        mine, found = ours.get(key), theirs.get(key)
        if mine != found:
            category.add({"key": key, "release": _plain(mine), other: _plain(found)})
    return category


def _plain(value: object) -> object:
    """`value` as JSON: a dataclass as its fields, a tuple as a list."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return list(value) if isinstance(value, tuple) else value


# ---------------------------------------------------------------- code index


def code_index_parity(
    release: Mapping[str, object], export: Mapping[str, object], *, examples: int
) -> dict[str, Category]:
    """The blocks of the release's code index document against a fresh export's, row by row.
    The `stamp` is not a block. Its counts and `content_hash` follow from the blocks, and its
    endpoint, export time and coverage build name each side's source."""
    found: dict[str, Category] = {}
    for block in sorted((release.keys() | export.keys()) - {"stamp"}):
        ours, theirs = _rows(block, release.get(block)), _rows(block, export.get(block))
        keys = sorted(ours.keys() | theirs.keys())
        category = Category(examples, compared=len(keys))
        for key in keys:
            mine, other = ours.get(key), theirs.get(key)
            if mine == other:
                continue
            if mine is None:
                difference: dict[str, object] = {"key": key, "side": "export_only"}
            elif other is None:
                difference = {"key": key, "side": "release_only"}
            else:
                difference = {"key": key, "side": "both", "fields": _moved_fields(mine, other)}
            category.add(difference)
        found[f"code_index.{block}"] = category
    return found


def _rows(block: str, value: object) -> dict[str, object]:
    """`value`'s rows by key: a list block's by `_ROW_KEYS` or canonical JSON, a mapping
    block's by its keys, any other value as one row. An absent block holds none."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): row for key, row in value.items()}
    if not isinstance(value, list):
        return {"": value}
    key = _ROW_KEYS.get(block)
    return {
        " ".join(str(row.get(name)) for name in key)
        if key and isinstance(row, Mapping)
        else json.dumps(row, sort_keys=True): row
        for row in value
    }


def _moved_fields(ours: object, theirs: object) -> list[str] | None:
    if not isinstance(ours, Mapping) or not isinstance(theirs, Mapping):
        return None
    return sorted(
        str(name) for name in ours.keys() | theirs.keys() if ours.get(name) != theirs.get(name)
    )


# ---------------------------------------------------------------- origin


@contextmanager
def origin_flag(on: bool) -> Iterator[None]:
    """`GRAPH_RELEASE_ORIGIN` on or off for the body of the `with` block, and
    `GRAPH_RELEASE_COVERAGE` on with it, which the origin flag needs. A serving process reads
    both per call (`graph_release.context.origin_release_enabled`), so the reads in the body
    take the path the flag picks. With the flag off, `GRAPH_RELEASE_COVERAGE` keeps the
    caller's value."""
    saved = {flag: os.environ.get(flag) for flag in (ORIGIN_FLAG, COVERAGE_FLAG)}
    if on:
        os.environ[ORIGIN_FLAG] = "1"
        os.environ[COVERAGE_FLAG] = "1"
    else:
        os.environ.pop(ORIGIN_FLAG, None)
    try:
        yield
    finally:
        for flag, value in saved.items():
            if value is None:
                os.environ.pop(flag, None)
            else:
                os.environ[flag] = value


@contextmanager
def serving(release: Release, catalogue: CatalogueSnapshot, *, on: bool) -> Iterator[None]:
    """The binding a request has with `GRAPH_RELEASE_ORIGIN` `on`: `catalogue` bound, and
    `release` too when the flag is on."""
    with (
        origin_flag(on),
        bound_catalogue(catalogue),
        bound_release(release) if on else nullcontext(),
    ):
        yield


def _origin(release: Release) -> ReleaseOrigin:
    if release.origin is None:
        raise ValueError(f"release {release.release_id} was opened without its origin view")
    return release.origin


def family_parity(release: Release, catalogue: CatalogueSnapshot, *, examples: int) -> Category:
    """`held_family` on every family key the catalogue build or the release holds, flag on
    against flag off: members in order, each property's values in order, the shared values
    and the total."""
    origin = _origin(release)
    keys = {*catalogue.families.arcs, *catalogue.families.un, *origin.families}

    def reads() -> dict[str, object]:
        return {json.dumps(list(key)): _summary(catalogue.held_family(key)) for key in keys}

    with serving(release, catalogue, on=False):
        off = reads()
    with serving(release, catalogue, on=True):
        on = reads()
    return _mapping_diff(on, off, examples=examples, other="catalogue")


def _summary(summary: FamilySummary | None) -> dict[str, object] | None:
    """`summary` as JSON, its mappings as pairs so their order counts."""
    if summary is None:
        return None
    return {
        "members": list(summary.members),
        "values": [[prop, list(values)] for prop, values in summary.values.items()],
        "shared": [[prop, value] for prop, value in summary.shared.items()],
        "total": summary.total,
        "total_name": summary.total_name,
        "un_only": summary.un_only,
    }


def arc_parity(
    release: Release, catalogue: CatalogueSnapshot, *, examples: int
) -> dict[str, Category]:
    """`held_arcs` of every DFO variable the catalogue build or the release holds, flag on
    against flag off, every row with its target's name and types (`origin.arcs`). Then each
    node those arcs name (`origin.arc_targets`): the names and types each side's arcs give
    it, and whether `arc_target` holds it, since `arc_target` and `arc_label` answer from the
    build alone and a target only the release names reads from the graph."""
    origin = _origin(release)
    variables = catalogue.arcs.dfo.keys() | origin.arcs.keys()

    def reads() -> dict[str, tuple[UnArc, ...] | None]:
        return {variable: catalogue.held_arcs(variable) for variable in variables}

    with serving(release, catalogue, on=False):
        off = reads()
    with serving(release, catalogue, on=True):
        on = reads()
    arcs = _mapping_diff(
        {variable: _arc_rows(on[variable]) for variable in variables},
        {variable: _arc_rows(off[variable]) for variable in variables},
        examples=examples,
        other="catalogue",
    )
    on_targets, off_targets = _target_rows(on), _target_rows(off)
    named = sorted(on_targets.keys() | off_targets.keys())
    targets = Category(examples, compared=len(named))
    for dcid in named:
        mine, theirs = on_targets.get(dcid, []), off_targets.get(dcid, [])
        held = catalogue.arc_target(dcid)
        if mine != theirs or held is None:
            targets.add(
                {
                    "key": dcid,
                    "release": mine,
                    "catalogue": theirs,
                    "arc_target": _plain(held),
                    "arc_label": catalogue.arc_label(dcid),
                }
            )
    return {"origin.arcs": arcs, "origin.arc_targets": targets}


def _arc_rows(arcs: tuple[UnArc, ...] | None) -> list[object] | None:
    return None if arcs is None else [_plain(arc) for arc in arcs]


def _target_rows(
    arcs: Mapping[str, tuple[UnArc, ...] | None],
) -> dict[str, list[list[object]]]:
    """Each node the arcs name, with the distinct names and types they give it, sorted."""
    found: dict[str, set[tuple[str | None, tuple[str, ...]]]] = {}
    for rows in arcs.values():
        for arc in rows or ():
            if arc.dcid is not None:
                found.setdefault(arc.dcid, set()).add((arc.name, arc.types))
    return {
        dcid: [[name, list(types)] for name, types in sorted(seen, key=str)]
        for dcid, seen in found.items()
    }


@dataclass(frozen=True)
class MemberWalk:
    """One member walk of a DFO family's five-tuple: every constraint property its members
    declare a slot left unbound, so the walk confirms each member, and no role key bound or
    the roles one member declares bound, so each role class is elected once."""

    five_tuple: FiveTuple
    constraint_properties: tuple[str, ...]
    role_keys: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return json.dumps(
            {
                **asdict(self.five_tuple),
                "constraint_properties": list(self.constraint_properties),
                "role_keys": list(self.role_keys),
            },
            sort_keys=True,
        )


def member_walks(origin: ReleaseOrigin) -> list[MemberWalk]:
    """The walks of every family of the release's DFO variables, in label order."""
    families: dict[FiveTuple, tuple[set[str], set[tuple[str, ...]]]] = {}
    for key, summary in origin.families.items():
        five_tuple = FiveTuple(
            population_type=key.population_type or None,
            measured_property=key.measured_property or None,
            stat_type=key.stat_type or None,
            measurement_qualifier=key.measurement_qualifier,
            measurement_denominator=key.measurement_denominator,
            denominator_unresolved=key.denominator_unresolved,
        )
        properties, role_sets = families.setdefault(five_tuple, (set(), {()}))
        properties.update(summary.values)
        for member in summary.members:
            role_sets.add(
                tuple(
                    sorted(
                        arc.dcid
                        for arc in origin.arcs[member]
                        if arc.property == _OBSERVATION_PROPERTIES and arc.dcid is not None
                    )
                )
            )
    walks = [
        MemberWalk(five_tuple, tuple(sorted(properties)), roles)
        for five_tuple, (properties, role_sets) in families.items()
        for roles in role_sets
    ]
    return sorted(walks, key=lambda walk: walk.label)


def walk_members(
    graph: MemberArcReader, walk: MemberWalk, *, readings: Mapping[str, AbsentReading]
) -> dict[str, object]:
    """`walk` over `graph` as the Datasets mint walks it, under `DEFAULT_SUBJECT_CEILING`
    and no deadline: its group as JSON, members and family in order, or the refusal it
    raised. The member cap is the subject ceiling, so a family past the serving cap
    (`DEFAULT_MEMBER_CAP`) is compared member by member; equal member counts refuse alike."""
    try:
        group = enumerate_members(
            five_tuple=walk.five_tuple,
            slots=dict.fromkeys(walk.constraint_properties, _UNBOUND),
            graph=graph,
            readings=readings,
            member_cap=DEFAULT_SUBJECT_CEILING,
            role_keys=walk.role_keys,
            subject_ceiling=DEFAULT_SUBJECT_CEILING,
        )
    except _WALK_REFUSALS as exc:
        return {"refused": type(exc).__name__, "detail": str(exc)}
    return {
        "members": [
            [member.stat_var, dict(member.constraints), list(member.declared_roles)]
            for member in group.members
        ],
        "family": list(group.family),
        "candidates_considered": group.candidates_considered,
        "candidate_cap": group.candidate_cap,
        "truncated": group.truncated,
        "proven_complete": group.proven_complete,
        "root_confirmed": group.root_confirmed,
        "full_read": group.full_read,
        "inconsistent_candidates": list(group.inconsistent_candidates),
        "undismissed_candidates": list(group.undismissed_candidates),
        "elected_open_role_keys": sorted(group.elected_open_role_keys),
    }


def unlisted_candidates(
    graph: MemberArcReader, five_tuple: FiveTuple, *, listed: Collection[str]
) -> tuple[str, ...]:
    """The variables of `five_tuple` that `graph`'s reverse read finds and `listed` does not
    hold: each subject of the walk's reverse query, up to `DEFAULT_SUBJECT_CEILING`, whose
    five-tuple arcs match `five_tuple` exactly."""
    if five_tuple.measured_property is None:
        return ()
    page = graph.node_subjects(
        five_tuple.measured_property,
        property="measuredProperty",
        filters=five_tuple_reverse_filters(
            population_type=five_tuple.population_type,
            stat_type=five_tuple.stat_type,
            measurement_qualifier=five_tuple.measurement_qualifier,
            measurement_denominator=five_tuple.measurement_denominator,
        ),
        max_results=DEFAULT_SUBJECT_CEILING,
    )
    others = [dcid for dcid in page.dcids if dcid not in listed]
    if not others:
        return ()
    arcs = graph.node_arcs_many(others, properties=_FIVE_TUPLE_ARCS)
    return tuple(
        dcid
        for dcid in others
        if (found := arcs.get(dcid)) is not None
        and five_tuple_arcs_match(found, **asdict(five_tuple))
    )


def unobserved_variables(*, series: pa.Table, edges: pa.Table) -> dict[str, FiveTuple]:
    """Each node a release's `edges` type `StatisticalVariable` that its `series` holds no
    observation of, with the five-tuple its edges give it. The origin view leaves them out, as
    the catalogue job does, though the mixer serves them to a member walk."""
    edge_columns = [
        edges[name].cast(pa.string()) for name in ("subject_id", "predicate", "object_id")
    ]
    subjects, predicates, objects = edge_columns
    typed = pc.and_(
        pc.equal(predicates, pa.scalar("typeOf")),
        pc.equal(objects, pa.scalar(DFO_VARIABLE_TYPE)),
    )
    declared = pc.unique(subjects.filter(typed))
    observed = pc.unique(series["variable_measured"].cast(pa.string()))
    unobserved = declared.filter(pc.invert(pc.is_in(declared, value_set=observed)))
    rows = pc.and_(
        pc.is_in(subjects, value_set=unobserved),
        pc.is_in(predicates, value_set=pa.array(_FIVE_TUPLE_ARCS)),
    )
    arcs: dict[str, dict[str, str]] = {str(dcid): {} for dcid in unobserved.to_pylist()}
    for subject, predicate, target in zip(
        *(cast("list[str]", column.filter(rows).to_pylist()) for column in edge_columns),
        strict=True,
    ):
        arcs[subject][predicate] = target
    return {
        dcid: FiveTuple(
            population_type=found.get("populationType"),
            measured_property=found.get("measuredProperty"),
            stat_type=found.get("statType"),
            measurement_qualifier=found.get("measurementQualifier"),
            measurement_denominator=found.get("measurementDenominator"),
        )
        for dcid, found in sorted(arcs.items())
    }


def member_parity(
    release: Release,
    catalogue: CatalogueSnapshot,
    *,
    live: MemberArcReader,
    readings: Mapping[str, AbsentReading],
    examples: int,
    workers: int = 1,
) -> tuple[Category, dict[str, object]]:
    """Every walk of every DFO family (`member_walks`) the flag on reads from the release
    (`ReleaseOrigin.holds_family`), over the release's reader against `live`, the reader the
    flag off walks, with `catalogue` bound. Also returns the routing counts and the two
    families the release-held rule cannot see: a variable `live` finds for a release-held
    five-tuple that is no DFO variable of the release (`unlisted_candidates`), split into
    the release's unobserved variables (`unobserved_variables`) and the rest, base Data
    Commons variables the catalogue does not list."""
    origin = _origin(release)
    walks = member_walks(origin)
    five_tuples = sorted({walk.five_tuple for walk in walks}, key=lambda found: str(found))
    with serving(release, catalogue, on=True):
        held = [
            five_tuple
            for five_tuple in five_tuples
            if origin.holds_family(five_tuple, catalogue=catalogue)
        ]
    walked = [walk for walk in walks if walk.five_tuple in set(held)]
    ours = {walk.label: walk_members(origin.reader, walk, readings=readings) for walk in walked}
    with serving(release, catalogue, on=False):
        theirs = dict(
            zip(
                (walk.label for walk in walked),
                _map(lambda walk: walk_members(live, walk, readings=readings), walked, workers),
                strict=True,
            )
        )
        found = _map(
            lambda five_tuple: unlisted_candidates(live, five_tuple, listed=origin.arcs),
            held,
            workers,
        )
    members = _mapping_diff(ours, theirs, examples=examples, other="live")

    unobserved = unobserved_variables(
        series=release.loaded.tables["series"], edges=release.loaded.tables["edges"]
    )
    held_set = set(held)
    candidates = {
        dcid: five_tuple for five_tuple, dcids in zip(held, found, strict=True) for dcid in dcids
    }
    unlisted = {dcid: tuple_ for dcid, tuple_ in candidates.items() if dcid not in unobserved}
    unobserved_live = sorted(dcid for dcid in candidates if dcid in unobserved)
    extra = {
        "member_routing": {
            "five_tuples": len(five_tuples),
            "release": len(held),
            "live": len(five_tuples) - len(held),
            "walks_compared": len(walked),
        },
        "blind_spots": {
            "unlisted_base_variables": {
                "count": len(unlisted),
                "examples": [
                    {"dcid": dcid, "five_tuple": asdict(unlisted[dcid])}
                    for dcid in sorted(unlisted)[:examples]
                ],
            },
            "unobserved_spanner_variables": {
                "count": len(unobserved),
                "in_release_held_families": sum(
                    1 for five_tuple in unobserved.values() if five_tuple in held_set
                ),
                "found_by_live_walks": len(unobserved_live),
                "examples": [
                    {"dcid": dcid, "five_tuple": asdict(unobserved[dcid])}
                    for dcid in (unobserved_live or sorted(unobserved))[:examples]
                ],
            },
        },
    }
    return members, extra


def _map[T, R](read: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """`read` over `items` in order, on `workers` threads, each under the caller's context,
    which carries the bound catalogue."""
    if workers <= 1:
        return [read(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, read, item) for item in items]
        return [cast("R", future.result()) for future in futures]


def skills_parity(
    release: Release,
    catalogue: CatalogueSnapshot,
    *,
    build: Callable[[], SkillsIndex],
    examples: int,
) -> dict[str, Category]:
    """The skills index `build` gives flag on against flag off: its roster (each skill's
    identity, files and issues, the roster order and the completeness flag), each skill's
    attachments, and its reach (nodes, StatVar count and each node's matches)."""
    with serving(release, catalogue, on=False):
        theirs = build()
    with serving(release, catalogue, on=True):
        ours = build()
    return {
        f"skills.{name}": _mapping_diff(view(ours), view(theirs), examples=examples, other="live")
        for name, view in (
            ("roster", _roster),
            ("attachments", _attachments),
            ("reach", _reach),
        )
    }


def _roster(index: SkillsIndex) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {
        skill_id: {
            "front_matter": asdict(entry.front_matter),
            "url": entry.url,
            "files": [[file.path, file.size, file.digest] for file in entry.files],
            "issues": [],
        }
        for skill_id, entry in index.skills.items()
    }
    for issue in index.issues:
        row = rows.setdefault(issue.skill_id or "", {"issues": []})
        cast("list[list[str]]", row["issues"]).append([issue.code, issue.message])
    return {
        **rows,
        "(index)": {
            "order": list(index.skills),
            "may_omit_published_skills": index.may_omit_published_skills,
        },
    }


def _attachments(index: SkillsIndex) -> dict[str, object]:
    return {
        skill_id: [
            [attachment.dcid, type(attachment.kind).__name__, _plain(attachment.kind)]
            for attachment in entry.attachments
        ]
        for skill_id, entry in index.skills.items()
    }


def _reach(index: SkillsIndex) -> dict[str, object]:
    return {
        **{
            skill_id: {
                "nodes": [[node.dcid, node.hop, node.via_property] for node in entry.reach.nodes],
                "excluded": entry.reach.excluded,
                "capped": entry.reach.capped,
                "statvar_count": entry.statvar_count,
                "statvar_count_truncated": entry.statvar_count_truncated,
            }
            for skill_id, entry in index.skills.items()
        },
        **{
            f"node:{dcid}": [asdict(match) for match in matches]
            for dcid, matches in index.node_to_skills.items()
        },
    }


def origin_parity_report(
    loaded: LoadedRelease,
    catalogue: CatalogueSnapshot,
    *,
    live: MemberArcReader,
    build_skills: Callable[[], SkillsIndex],
    readings: Mapping[str, AbsentReading],
    examples: int = _DEFAULT_EXAMPLES,
    workers: int = 1,
) -> dict[str, object]:
    """The report on `loaded`'s origin view against the catalogue build `catalogue` of the
    same import and the reads of `live`, the graph the flag off walks members on.
    `build_skills` builds the skills index under whatever binding it is called in. A release
    whose origin view does not build counts in category `views`, and nothing else is compared."""
    views = Category(examples, compared=1)
    categories: dict[str, Category] = {"views": views}
    extra: dict[str, object] = {}
    try:
        release = Release(loaded=loaded, origin=release_origin(loaded))
    except ViewRefused as exc:
        views.add({"view": "origin", "refused": str(exc)})
    else:
        categories["origin.families"] = family_parity(release, catalogue, examples=examples)
        categories |= arc_parity(release, catalogue, examples=examples)
        categories["origin.members"], extra = member_parity(
            release, catalogue, live=live, readings=readings, examples=examples, workers=workers
        )
        categories |= skills_parity(release, catalogue, build=build_skills, examples=examples)
    differences = sum(category.differences for category in categories.values())
    return {
        "outcome": "differences" if differences else "parity",
        "differences": differences,
        "release": _release_summary(loaded.manifest),
        "origin_catalogue": _catalogue_summary(catalogue.manifest),
        **extra,
        "categories": {name: category.to_json() for name, category in sorted(categories.items())},
    }


def _catalogue_summary(manifest: CatalogueManifest) -> dict[str, object]:
    return {
        "build_id": manifest.build_id,
        "identity": manifest.identity,
        "built_at": manifest.built_at.isoformat(),
        "workflow_execution_id": (
            manifest.ingestion.workflow_execution_id if manifest.ingestion else None
        ),
    }


# ---------------------------------------------------------------- report


def parity_report(
    release: LoadedRelease,
    published: LoadedCoverage,
    published_rows: pa.Table,
    export: Mapping[str, object],
    *,
    examples: int = _DEFAULT_EXAMPLES,
) -> dict[str, object]:
    """The report on `release` against the coverage build `published` and the fresh code index
    `export`. The release's serving views that do not build count in category `views`."""
    views = Category(examples, compared=2)
    categories: dict[str, Category] = {"views": views}
    try:
        coverage = release_coverage_index(release)
    except ViewRefused as exc:
        views.add({"view": "coverage", "refused": str(exc)})
    else:
        categories |= coverage_parity(
            release, coverage, published, published_rows, examples=examples
        )
    try:
        release_code_indexes(release)
    except ViewRefused as exc:
        views.add({"view": "code_index", "refused": str(exc)})

    document = read_document(release.blobs[CODE_INDEX_PART])
    categories |= code_index_parity(document, export, examples=examples)
    differences = sum(category.differences for category in categories.values())
    return {
        "outcome": "differences" if differences else "parity",
        "differences": differences,
        "release": _release_summary(release.manifest),
        "coverage_build": _coverage_summary(published.manifest),
        "code_index": {
            "release_content_hash": content_hash(_blocks(document)),
            "export_content_hash": content_hash(_blocks(export)),
            "export_stamp": export.get("stamp"),
        },
        "categories": {name: category.to_json() for name, category in sorted(categories.items())},
    }


def _blocks(document: Mapping[str, object]) -> dict:
    return {name: value for name, value in document.items() if name != "stamp"}


def _release_summary(manifest: Manifest) -> dict[str, object]:
    return {
        "release_id": manifest.release_id,
        "import_execution_id": manifest.import_execution_id,
        "read_timestamp": manifest.read_timestamp,
        "dc_serving_revision": manifest.dc_serving_revision,
        "declarations_hash": manifest.declarations_hash,
        "builder_code_hash": manifest.builder_code_hash,
        "checkout_builder_code_hash": builder_code_hash(),
    }


def _coverage_summary(manifest: CoverageManifest) -> dict[str, object]:
    return {
        "build_id": manifest.build_id,
        "stamp": manifest.stamp,
        "built_at": manifest.built_at,
        "workflow_execution_id": (
            manifest.ingestion.workflow_execution_id if manifest.ingestion else None
        ),
    }


# ---------------------------------------------------------------- main


def _run_coverage(args: argparse.Namespace, local_dir: Path) -> dict[str, object]:
    release = open_release(args.release_uri, local_dir=local_dir, parts=_RELEASE_PARTS)
    manifest = release.manifest
    pointer = read_published_pointer(args.coverage_uri)
    if pointer is None:
        raise Refused("coverage_unreadable", f"no coverage pointer under {args.coverage_uri}")
    published = load_coverage(args.coverage_uri, pointer=pointer)
    check_same_import(manifest, published.manifest)
    check_declarations(manifest)
    admin = CloudRunAdmin(service=args.datacommons_service)

    def serving() -> str:
        return admin.serving().token

    try:
        check_serving_revision(manifest, serving)
        reader = export_reader(f"{args.graph_base}/core/api", os.environ.get("SEARCH_GRAPH_TOKEN"))
        try:
            export = export_index(
                reader,
                graph_endpoint=args.graph_base,
                now=datetime.now(UTC),
                coverage=coverage_join(
                    published,
                    datasets={dataset.provenance for dataset in DATASETS} | set(LISTED_DATASETS),
                ),
                listed_datasets=LISTED_DATASETS,
            )
        except ExportRefused as exc:
            raise Refused("export_refused", str(exc)) from exc
        finally:
            reader.close()
        check_serving_revision(manifest, serving)
    finally:
        admin.close()
    prefix = join_uri(format_root(args.coverage_uri), VERSIONS_PREFIX, pointer.build_id)
    rows = read_table(prefix, published.manifest)
    return parity_report(release, published, rows, export, examples=args.examples)


def _run_origin(args: argparse.Namespace, local_dir: Path) -> dict[str, object]:
    # Search's settings require its mixer root at import, which is the one this run reads.
    os.environ.setdefault("SEARCH_GRAPH_BASE", args.graph_base)
    # Imported here: the search engine and the skills build, which the coverage gate does not read.
    from code_index.shared import get_index
    from origin_catalogue.artifact import load_catalogue
    from origin_catalogue.graph import build_graph_client
    from origin_catalogue.service import CatalogueSettings
    from search.engine.clients.graph_client import LiveGraphClient
    from skills_index.build import build_index

    loaded = open_release(args.release_uri, local_dir=local_dir, parts=_ORIGIN_PARTS)
    with origin_flag(True):
        # Loaded as a serving process loads it with the flag on, with its base summaries.
        catalogue = load_catalogue(
            args.origin_catalogue_uri, un_system_enabled=CatalogueSettings().un_system_enabled
        )
    check_same_import(loaded.manifest, catalogue.manifest, artifact="catalogue")
    index = get_index()
    readings = {
        code_system: declaration.absent_reading
        for code_system, declaration in index.declarations.items()
    }
    token = os.environ.get("SEARCH_GRAPH_TOKEN")
    admin = CloudRunAdmin(service=args.datacommons_service)

    def serving_token() -> str:
        return admin.serving().token

    def build_skills() -> SkillsIndex:
        graph = LiveGraphClient(base=args.graph_base, token=token or "")
        try:
            # LiveGraphClient satisfies SkillsGraph structurally, but the two
            # Protocols name separately defined SubjectsPage classes.
            return build_index(graph=graph)  # ty: ignore[invalid-argument-type]
        finally:
            graph.close()

    try:
        check_serving_revision(loaded.manifest, serving_token)
        live = build_graph_client(base_url=f"{args.graph_base}/core/api", token=token)
        try:
            report = origin_parity_report(
                loaded,
                catalogue,
                live=live,
                build_skills=build_skills,
                readings=readings,
                examples=args.examples,
                workers=args.workers,
            )
        finally:
            live.close()
        check_serving_revision(loaded.manifest, serving_token)
    finally:
        admin.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    gates = parser.add_subparsers(dest="gate", required=True)
    coverage = gates.add_parser("coverage", help="the GRAPH_RELEASE_COVERAGE gate")
    coverage.add_argument(
        "--coverage-uri", required=True, help="the environment's COVERAGE_ARTIFACT_URI"
    )
    origin = gates.add_parser("origin", help="the GRAPH_RELEASE_ORIGIN gate")
    origin.add_argument(
        "--origin-catalogue-uri",
        default=os.environ.get("ORIGIN_CATALOGUE_URI"),
        help="the environment's ORIGIN_CATALOGUE_URI (default ORIGIN_CATALOGUE_URI)",
    )
    origin.add_argument(
        "--workers", type=int, default=8, help="threads reading the mixer (default 8)"
    )
    for gate in (coverage, origin):
        gate.add_argument(
            "--release-uri", required=True, help="the environment's GRAPH_RELEASE_URI"
        )
        gate.add_argument(
            "--graph-base",
            default=os.environ.get("SEARCH_GRAPH_BASE"),
            help="the mixer root, without /core/api (default SEARCH_GRAPH_BASE)",
        )
        gate.add_argument(
            "--datacommons-service",
            default=os.environ.get("DATACOMMONS_SERVICE_NAME"),
            help="the DC service as projects/<p>/locations/<r>/services/<s> "
            "(default DATACOMMONS_SERVICE_NAME)",
        )
        gate.add_argument("--local-dir", type=Path, help="where to keep the release's parts")
        gate.add_argument("--examples", type=int, default=_DEFAULT_EXAMPLES)
        gate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.graph_base or not args.datacommons_service:
        parser.error("--graph-base and --datacommons-service are required")
    if args.gate == "origin" and not args.origin_catalogue_uri:
        parser.error("--origin-catalogue-uri is required")
    run = _run_coverage if args.gate == "coverage" else _run_origin

    try:
        with tempfile.TemporaryDirectory(prefix="graph-release-parity-") as scratch:
            report = run(args, args.local_dir or Path(scratch))
    except Refused as exc:
        report = {"outcome": "refused", "refused": {"code": exc.code, "detail": exc.detail}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": report["outcome"], "differences": report.get("differences")}))
    sys.exit({"parity": 0, "differences": 1, "refused": 2}[str(report["outcome"])])


if __name__ == "__main__":
    main()
