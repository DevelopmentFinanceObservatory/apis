"""The analysis runner's roots, read from this service's own stores (`LiveRootSource`).

A root is read as data-mcp's `get_data` reads a frame, so a recipe the runner stores names its
roots as the analysis tools name theirs. A mint record's `options` are the bare frame-layout
Parquet `{"format": "parquet", "bundle": false, "layout": "frame"}`, with `columns` when the
root asks for enrichment, `without_party_totals: true` when it leaves party totals out, and
`groupby` when it reads the frame summed.

- A root read with any of those three is rendered as a definition read renders that mint from
  the version's stored frame (`current.head_rendering`). A `groupby` sum or a party-totals cut
  that changes the frame is a version of its own (`render.DerivedVersion`), and that version is
  the root's dataset, as it is the `dataset` of data-mcp's mint record.
  Once a read has recorded that version, a later read of the same dataset version with the
  same minting options under the same code index and render format reads it with no render
  (`LiveRootSource._recorded_rendering`).
- With no `columns` the frame is the dataset version's stored frame, its canonical frame (or
  frame1 document).
- With `columns` the frame is the dataset version's frame-layout file of those columns, decoded
  as data-mcp decodes it: every column outside the grammar's reserved set is a role key, an
  empty enrichment cell reads `untyped`, and a text `unit` cell holds a unit ref or the JCS text
  of a converted unit.

The frame is named by the pair `frame_ref_for` derives from the columns, the groupby and the
party-totals cut, and for a frame-layout file from the render format its cells were written
under (`versions.RENDER_FORMAT`), so two renderings of one version are two frames.

A root that leaves `columns` to the default is read with the version's default enrichment
(`default_enrichment`): the root asks the mint for `enrichment: "default"`, which resolves the
columns from the roles the version's frame holds, and its record keeps the resolved `columns`
(`record_options`). The verbs read them to keep group totals out of a sum or rank and to rank
places at one level.

A request root that states no `groupby`, under a step other than `filter`, and whose
version's frame holds more than one StatVar with data (`description_record.summed_refs`) is
read summed (`implied_groupby`), unless a step over it names `stat_var`: its options gain
`groupby` `{by: <"time_period" and the summable axes its steps name, sorted>, op: "sum"}`, and
`codes`, the code dcids the filters over it keep on each axis `by` keeps, when they keep some
(`analysis.PlannedRoot`), so its record, the recipe's normal form and the format 2 ref state the
sum. When the sum refuses
`rollup_not_summable`, inline or as a queued rebuild that failed with that code, the root is
read unsummed, records the refusal (`ResolvedRoot.sum_refused`) and logs one
`datasets_analysis_root_sum_refused` line. A root whose groupby summed StatVars names them
(`ResolvedRoot.stat_vars_summed`).

A Spec root is served as the definition door serves a registered id (`doors.definition`): its
newest version of the rendering, rebuilt when its head is stale, in one serve, summed as the
stored head's trace implies so no other rendering is made. A dataset root under `latest` is
served the same way with the rendering it records, so it is read by its rendering, since its
version alone can record two renderings that the newest data splits. A request reads one
version of a dataset: a later root of a dataset an earlier root served renders from the
version that serve read. A rebuild
over the inline budget answers `AnalysisPending` naming its operation, and a rebuild that failed
and will not be retried refuses as the refresh door refuses it (`RefreshRebuildFailed`).
`newest_spec_version` names the dataset version a Spec root reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rfc8785
from dfo_frame_grammar import (
    Frame,
    build_frame_from_arrow,
    canonical_frame_schema,
    map_units,
    unit_cell_array,
)
from dfo_frame_grammar.grammar import (
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    UnitRequest,
    unit_cell_from_json,
)

from artifact_store import Tier
from artifact_store.errors import ArtifactNotFoundError
from code_index.query import CodeIndex
from code_index.types import summable_axes
from dataset_store.snapshot import (
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
    read_servable_version,
)
from datasets_api.analysis import (
    STAT_VAR_PREFIX,
    STAT_VAR_WORD,
    AnalysisPending,
    AnalysisTooLarge,
    DatasetRoot,
    LocatedRoot,
    ResolvedRoot,
    RootExpired,
    RootNotStored,
    RootRendering,
    SpecRoot,
    recorded_dataset_root,
)
from datasets_api.current import (
    DefinitionDeps,
    UnstoredFile,
    UnstoredRendering,
    head_rendering,
    load_stored,
    render_unstored,
)
from datasets_api.default_enrichment import (
    KIND_COLUMNS,
    coverage_roles,
    with_default_enrichment,
)
from datasets_api.derived import root_rendering
from datasets_api.description_record import summed_refs
from datasets_api.doors.definition import serve_spec, served_spec
from datasets_api.doors.refresh import RefreshRebuildFailed
from datasets_api.doors.snapshot import servable_snapshot_manifest
from datasets_api.groupby import adds_stat_vars, selected_contributors
from datasets_api.manifest_models import ArtifactEntryModel, CoverageModel, Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.operations.submit import OPERATION_SUBMIT_RETRY_AFTER_MS, OperationSubmitted
from datasets_api.recipe_store import RecipeReader
from datasets_api.recorded_rendering import MintingOptions, mintings_rendered_under
from datasets_api.render import (
    check_options_supported,
    is_statistics_rendering,
    refuse_frame1_past_decode_bound,
    refuse_past_fold_ceiling,
    resolved_rendering_keys,
    servable_frame_location,
)
from datasets_api.store_errors import (
    read_stored_frame,
    stored_frame_distinct,
    stored_frame_row_count,
)
from datasets_api.timings import phase
from datasets_api.versions import RENDER_FORMAT, rendering_stamp
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.place_type import (
    UNTYPED,
    enrichment_role,
)
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from materialise.reshape import RollupNotSummableError
from series_coverage.shared import CoverageSnapshot
from spec_registry.reader import StoredDefinition

logger = logging.getLogger(__name__)

# The bare frame-layout Parquet data-mcp mints a frame with.
_BASE_OPTIONS: Mapping[str, Any] = {"format": "parquet", "bundle": False, "layout": "frame"}
_RESERVED_COLUMNS = frozenset(
    {"stat_var", "time_period", "value", "unit", "measurementMethod", "provenance", "status"}
)
_OPTIONAL_COLUMNS = frozenset({UNIT_COLUMN, "measurementMethod", "provenance"})
# The recipe layout holds every member a converted cell can carry. The frame build narrows
# a column whose cells carry no conversion member to the layout without them.
_UNIT_TYPE = canonical_frame_schema((UNIT_COLUMN,), recipe=True).field(UNIT_COLUMN).type


def requests_kinds(columns: Sequence[str]) -> bool:
    """Whether `columns` asks for a flow party's kind, class or whole cells."""
    return not set(KIND_COLUMNS).isdisjoint(columns)


def frame_ref_for(
    *,
    spec_id: str,
    content_version: str,
    columns: Sequence[str],
    groupby: Mapping[str, Any] | None = None,
    without_party_totals: bool = False,
    enrichment_stamp: str | None,
) -> FrameRef:
    """The pair a frame of `columns`, `groupby` and the party-totals cut read from the version
    `{spec_id, content_version}` is named by: the version's own pair with none of them, else
    `cv_` and 32 hex digits of the SHA-256 of the RFC 8785 form of `{"columns": sorted,
    "content_version"}`, with `groupby` when set, its `by` sorted, `"without_party_totals":
    true` when the cut is asked for, the `enrichment_stamp` of its file when the columns ask
    for kinds, and `"render_format": versions.RENDER_FORMAT` when `columns` is not empty.
    data-mcp's `frame_materialisation.frame_ref_for` names frames from the columns, the
    content version and the groupby alone.

    The pair names what a recipe replays: a cut that leaves nothing out serves its base
    version, and the frame read with it is still named apart from the frame read without it,
    since the two replay differently once the data holds party totals. A root with `columns`
    reads its frame-layout file, whose decoded `stat_var_<code_system>` and party-kind cells
    are the render format's, so a frame rendered under another format is another frame. The
    other roots read a stored frame, which a format change gives a new `content_version`
    (`render.DerivedVersion`) or leaves as it is."""
    return frame_ref_under(
        spec_id=spec_id,
        content_version=content_version,
        columns=columns,
        groupby=groupby,
        without_party_totals=without_party_totals,
        enrichment_stamp=enrichment_stamp,
        render_format=RENDER_FORMAT,
    )


def frame_ref_under(
    *,
    spec_id: str,
    content_version: str,
    columns: Sequence[str],
    groupby: Mapping[str, Any] | None,
    without_party_totals: bool,
    enrichment_stamp: str | None,
    render_format: str | None,
) -> FrameRef:
    """The pair `frame_ref_for` names a frame by when frame-layout files are written in
    `render_format`. `None` names it as a root was named before its name carried a render
    format, with no `render_format` member, so a stored recipe's root names the format of the
    file it read (`recipe_replay.StoredRootSource`)."""
    if not columns and groupby is None and not without_party_totals:
        return FrameRef(spec_id=spec_id, content_version=content_version)
    preimage: dict[str, Any] = {"columns": sorted(columns), "content_version": content_version}
    if groupby is not None:
        preimage["groupby"] = {**groupby, "by": sorted(groupby["by"])}
    if without_party_totals:
        preimage["without_party_totals"] = True
    if enrichment_stamp is not None and requests_kinds(columns):
        preimage["enrichment_stamp"] = enrichment_stamp
    if columns and render_format is not None:
        preimage["render_format"] = render_format
    digest = hashlib.sha256(rfc8785.dumps(preimage)).hexdigest()[:32]
    return FrameRef(spec_id=spec_id, content_version=f"cv_{digest}")


def mint_options(rendering: RootRendering) -> dict[str, Any]:
    """The mint `options` a root of `rendering` is read with. A rendering that leaves `columns`
    to the default asks for `enrichment: "default"` in their place, which `record_options`
    resolves against the version read, as the mint resolves it.

    `columns` are sorted and each listed once, the order `frame_ref_for` names the frame by, so
    the frame pair, the frame-layout file, the record and the recipe's normal form all see one
    order, and two roots that list the same columns in another order are one root. `groupby.by`
    is sorted for the same reason."""
    options: dict[str, Any] = dict(_BASE_OPTIONS)
    if rendering.columns is None:
        options["enrichment"] = "default"
    elif rendering.columns:
        options["columns"] = sorted(set(rendering.columns))
    if rendering.without_party_totals:
        options["without_party_totals"] = True
    if rendering.groupby is not None:
        groupby = rendering.groupby.model_dump(mode="json", exclude_none=True)
        options["groupby"] = {**groupby, "by": sorted(groupby["by"])}
    return options


def record_options(options: Mapping[str, Any], *, coverage: CoverageModel) -> dict[str, Any]:
    """`options` as a root's record keeps them once read from a version of `coverage`: an
    `enrichment: "default"` replaced in place by the `columns` it resolves to
    (`default_enrichment.with_default_enrichment`), or dropped when it resolves to none."""
    resolved: dict[str, Any] = {}
    for key, value in options.items():
        if key != "enrichment":
            resolved[key] = value
        elif columns := with_default_enrichment(None, roles=coverage_roles(coverage)):
            resolved["columns"] = columns
    return resolved


def implied_groupby(
    stat_vars: Sequence[str],
    *,
    path_words: frozenset[str],
    code_index: CodeIndex,
    codes: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any] | None:
    """The `groupby` a root read with no stated one is read with, from the StatVars with data
    in its version, `stat_vars` as bare dcids (`LiveRootSource._contributing`), and the words
    the steps over it name: none when the words name `stat_var` or there are fewer than two
    StatVars, else `{by, op: "sum"}`, `by` `time_period` and the summable axes those StatVars
    decode (`code_index` declarations) that the words name, as the axis or as its
    `stat_var_<axis>` column, sorted as `mint_options` sorts a stated `by`: the groupby a
    caller following the refusal's summed read states. The datasets door sums every other
    summable axis away, and `stat_var` with them (conformance C120).

    `codes`, the code dcids the filters over the root keep on each axis
    (`analysis.PlannedRoot`), becomes the groupby's `codes` on each axis `by` keeps, sorted and
    each once as `GroupbyOptions` holds them, so the sum reads only the StatVars whose rows
    those filters keep (conformance C120b); a groupby keeping none of their axes has none, and
    neither has one selecting none of `stat_vars` (`selected_contributors`), whose sum would be
    empty where the filter over the unselected sum lists the values its column holds
    (`unmatched_breakdown_values`)."""
    if STAT_VAR_WORD in path_words:
        return None
    if len(stat_vars) < 2:
        return None
    summable = summable_axes(code_index.declarations)
    axes: set[str] = set()
    for dcid in stat_vars:
        if (row := code_index.statvar_by_dcid(dcid)) is not None:
            axes.update(summable.intersection(row.constraint_properties))
    named = {word.removeprefix(STAT_VAR_PREFIX) for word in path_words} & axes
    groupby: dict[str, Any] = {"by": sorted({TIME_PERIOD_COLUMN, *named}), "op": "sum"}
    selected = {axis: sorted(set(kept)) for axis, kept in (codes or {}).items() if axis in named}
    if selected and selected_contributors(stat_vars, index=code_index, codes=selected):
        groupby["codes"] = dict(sorted(selected.items()))
    return groupby


def _sum_refused(refused: RollupNotSummableError) -> dict[str, str]:
    return {"axis": refused.axis, "precondition": refused.precondition}


_SUM_REFUSED_EVENT = "datasets_analysis_root_sum_refused"
"""The `event` of the line logged when the datasets door refuses a sum the planner implied for
an analysis root, which is then read unsummed."""


def _log_sum_refused(
    *,
    spec_id: str,
    content_version: str,
    by: Mapping[str, Any],
    refused: Mapping[str, str],
    operation_id: str | None = None,
) -> None:
    logger.info(
        "analysis root read unsummed spec_id=%s content_version=%s axis=%s precondition=%s",
        spec_id,
        content_version,
        refused["axis"],
        refused["precondition"],
        extra={
            JSON_FIELDS_ATTR: {
                "event": _SUM_REFUSED_EVENT,
                "spec_id": spec_id,
                "content_version": content_version,
                "by": list(by["by"]),
                "axis": refused["axis"],
                "precondition": refused["precondition"],
                "operation_id": operation_id,
            }
        },
    )


def summed_count(
    version: StoredVersion | None, *, rendered: int | None, contributors: int, adds: bool = True
) -> int:
    """How many StatVars a groupby rendering summed into its version, whose record is
    `version`: the count a stored `stat_vars_summed` warning states (`rendered`), else
    `contributors`, the StatVars with rows in the dataset version it renders that its
    `groupby.codes` selection keeps, the ones the renderer sums
    (`GroupbyCollapseRendering.contributing_member_count`); 0 under two. 0 too
    when the record says the frame summed nothing, as for a version with no base version or a
    derived one whose `frame_summed` is false, and when the groupby `adds` nothing, a re-key
    keeping each StatVar in a tuple of its own (`groupby.adds_stat_vars`). A missing record or
    an unrecorded `frame_summed` is unknown and reads as summed."""
    if not adds or (
        version is not None
        and (version.base_content_version is None or version.frame_summed is False)
    ):
        return 0
    count = rendered if rendered is not None else contributors
    return count if count > 1 else 0


def _listed_base(options: Mapping[str, Any], base: str) -> str | None:
    """The dataset version a root read with `options` names as its rendering's base: `base`,
    the version a groupby sums, for a sum of every StatVar, which keeps the dataset's periods
    and entities. A party-totals cut, and a sum whose groupby selects `codes`, can each leave a
    period or an entity out, so a rendering that cuts or selects names no base and is matched
    to no dataset version (conformance C121f)."""
    groupby = options.get("groupby")
    if groupby is None or options.get("without_party_totals") is True or groupby.get("codes"):
        return None
    return base


def _with_summed(root: ResolvedRoot, count: int, named: Sequence[str]) -> ResolvedRoot:
    """`root` with the `count` StatVars its groupby summed, `named` as far as its trace names
    them."""
    if not count:
        return root
    return replace(root, summed_count=count, stat_vars_summed=tuple(named))


def _rendered_count(manifest: Manifest) -> int | None:
    """The count of StatVars with rows the renderer summed into a groupby rendering, as the
    `stat_vars_summed` warning a version stored while collapses raised one carries, `None`
    when it carries none."""
    return next(
        (
            warning.contributing_member_count
            for warning in manifest.warnings
            if warning.code == _SUMMED_WARNING
        ),
        None,
    )


_SUMMED_WARNING = "stat_vars_summed"
_DCID = "dcid:"


def _counts_recorded(trace: Mapping[str, Any]) -> bool:
    """Whether every StatVar `trace` resolved records its observation count."""
    entries = trace.get("resolved_stat_vars") or ()
    return all(isinstance(entry.get("observation_count"), int) for entry in entries)


# The precondition a queued rebuild's `rollup_not_summable` names when its operation records
# no refusal body to read it from.
_UNRECORDED = "unrecorded"


def _rebuild_sum_refused(failed: RefreshRebuildFailed) -> dict[str, str]:
    """The `{axis, precondition}` of the `rollup_not_summable` a queued rebuild met, taken from the axis of
    its recorded issue and its recorded detail, which names the precondition, as far as the
    operation records them."""
    refusal = failed.refusal
    issues = (refusal.issues or []) if refusal is not None else []
    axis = next((issue.axis for issue in issues if issue.axis), None)
    detail = refusal.detail if refusal is not None else None
    return {"axis": axis or _UNRECORDED, "precondition": detail or _UNRECORDED}


@dataclass(frozen=True)
class _FrameLayoutFile:
    """A root read from its frame-layout file `artifact`."""

    artifact: ArtifactEntryModel


@dataclass(frozen=True)
class _LocatedFrame:
    """A root read from its version's stored frame at `location`, as `locate` found it."""

    location: StoredArtifactLocation


@dataclass(frozen=True)
class _UnstoredFrameLayout:
    """A root read from the frame-layout bytes `data` a verification rendered in memory."""

    data: bytes


@dataclass(frozen=True)
class _UnstoredFrame:
    """A root read from the canonical frame `file` a verification rendered in memory."""

    file: UnstoredFile


_Handle = _FrameLayoutFile | _LocatedFrame | _UnstoredFrameLayout | _UnstoredFrame
"""What `read` reads. `resolve` gives a root read from its stored frame no handle, so a root
`locate` has not located holds none `read` accepts."""


def _data_artifact(manifest: Manifest) -> ArtifactEntryModel | None:
    return next((a for a in manifest.artifacts if not is_statistics_rendering(a.rendering)), None)


def _with_untyped_enrichment(table: pa.Table) -> pa.Table:
    """`table` with each empty or null enrichment cell written `untyped`."""
    roles = [name for name in table.schema.names if name not in _RESERVED_COLUMNS]
    for name in roles:
        if enrichment_role(name, roles=roles) is None:
            continue
        index = table.schema.get_field_index(name)
        column = table.column(index).combine_chunks()
        if pa.types.is_null(column.type):
            filled: pa.Array = pa.array([UNTYPED] * len(column), pa.string())
        else:
            empty = pc.or_kleene(
                pc.is_null(column), pc.equal(column, pa.scalar("", type=column.type))
            )
            filled = pc.if_else(empty, UNTYPED, column)
        table = table.set_column(index, name, filled)
    return table


def _unit_struct(text: object) -> object:
    """A text `unit` cell as the frame's tagged unit struct value."""
    if text is None:
        return None
    cell = cast(str, text)
    unit = (
        unit_cell_from_json(json.loads(cell)) if cell.startswith("{") else unit_cell_from_json(cell)
    )
    if cell.startswith("{") and not isinstance(unit, UnitRequest):
        raise ValueError(f"unit text {cell[:80]!r} is not a converted-unit object")
    return unit_cell_array(unit, 1)[0].as_py()


def frame_from_frame_layout(data: bytes) -> Frame:
    """The frame a frame-layout Parquet file of this service's holds. See the module docstring."""
    table = _with_untyped_enrichment(pq.read_table(pa.BufferReader(data)))
    for index, schema_field in enumerate(table.schema):
        column = table.column(index)
        if (
            schema_field.name in _OPTIONAL_COLUMNS
            and len(column) > 0
            and column.null_count == len(column)
            and not (schema_field.name == UNIT_COLUMN and pa.types.is_struct(schema_field.type))
        ):
            table = table.set_column(index, schema_field.name, pa.nulls(len(column), pa.string()))
    if UNIT_COLUMN in table.schema.names:
        index = table.schema.get_field_index(UNIT_COLUMN)
        if pa.types.is_string(table.schema.field(index).type):
            decoded = map_units(table.column(index), _unit_struct, result_type=_UNIT_TYPE)
            table = table.set_column(index, UNIT_COLUMN, decoded)
    if STATUS_COLUMN in table.schema.names:
        table = table.drop_columns(STATUS_COLUMN)
    role_keys = [name for name in table.schema.names if name not in _RESERVED_COLUMNS]
    return build_frame_from_arrow(table, role_keys=role_keys)


def _root(
    *,
    spec_id: str,
    content_version: str,
    options: Mapping[str, Any],
    artifact: ArtifactEntryModel | None,
    rows: int,
    source: object,
    base_content_version: str | None = None,
) -> ResolvedRoot:
    """The root read at `{spec_id, content_version}` with `options`, its frame named by
    `frame_ref_for` and `artifact`'s enrichment stamp, read through `source`, rendered from the
    dataset version `base_content_version`."""
    columns = tuple(options.get("columns") or ())
    stamp = artifact.enrichment_stamp if artifact is not None else None
    frame = frame_ref_for(
        spec_id=spec_id,
        content_version=content_version,
        columns=columns,
        groupby=options.get("groupby"),
        without_party_totals=options.get("without_party_totals") is True,
        enrichment_stamp=stamp,
    )
    dataset = FrameRef(spec_id=spec_id, content_version=content_version)
    record: dict[str, Any] = {
        "schema_version": 1,
        "frame": frame.model_dump(),
        "dataset": dataset.model_dump(),
        "options": dict(options),
        "scope": {},
    }
    if stamp is not None and requests_kinds(columns):
        record["enrichment_stamp"] = stamp
    return ResolvedRoot(
        frame=frame,
        dataset=dataset,
        record=record,
        rendering=root_rendering(options),
        rows=rows,
        source=source,
        base_content_version=base_content_version,
    )


class LiveRootSource:
    """`analysis.RootSource` over the Datasets surface's stores, for one request.

    `deps` is the request's retrieval view (`doors.definition.request_definition_deps`).
    `render` false reads a dataset root's rendering where its file is stored, and renders one
    whose file has aged out in memory from the version's stored frame (`current.render_unstored`),
    storing nothing, so a verification (`analysis.verify_analysis`) writes nothing and still
    reads a version whose stored frame is held. `max_root_rows` is then the most rows the roots
    this source has located hold together: a rendering that would take them past it is refused
    `AnalysisTooLarge` (`analysis_root_rows`) before it is rendered, counted at its version's
    stored frame's rows, which bound the rendering's. A root that resolves and then fails
    `locate` counts nothing, since nothing of it is read."""

    def __init__(
        self,
        *,
        deps: DefinitionDeps,
        recipes: RecipeReader,
        render: bool = True,
        max_root_rows: int | None = None,
    ) -> None:
        self._deps = deps
        self._recipes = recipes
        self._render = render
        self._max_root_rows = max_root_rows
        self._located_rows = 0
        # The version each served definition's first newest root read, by its spec id, with the
        # dataset version the serve read it from, `None` for its record's base (`_first_read`).
        # Written once per spec id, so every root of that dataset in the request reads one
        # dataset version.
        self._newest_versions: dict[str, tuple[str, str | None]] = {}
        # The StatVars with rows in each old-trace dataset version, by (spec id, version).
        self._stat_vars: dict[tuple[str, str], list[str]] = {}

    @property
    def code_index(self) -> CodeIndex:
        """The request's code index, which breakdown filter values resolve through."""
        return self._deps.code_index

    def _refuse_rendering_past_bound(self, rows: int) -> None:
        """Refuse a rendering in memory of a stored frame of `rows` rows that would take the
        roots located past `max_root_rows`.

        Raises:
            AnalysisTooLarge: `analysis_root_rows`.
        """
        cap = self._max_root_rows
        held = self._located_rows + rows
        if cap is not None and held > cap:
            raise AnalysisTooLarge(ceiling_kind="analysis_root_rows", count=held, cap=cap)

    def resolve(
        self,
        root: SpecRoot | DatasetRoot,
        *,
        latest: bool,
        trace_id: str | None,
        path_words: frozenset[str] | None = None,
        codes: Mapping[str, Sequence[str]] | None = None,
    ) -> ResolvedRoot:
        """The version `root` is read at (`RootSource.resolve`). A stated groupby the code
        index cannot serve refuses first (`check_options_supported`), before any version,
        stored rendering or answer is looked up by it.

        Raises:
            QueryOptionsInvalid: As `check_options_supported` refuses `root`'s rendering.
        """
        deps = self._deps
        if root.rendering.groupby is not None:
            check_options_supported(
                MintOptions.model_validate(mint_options(root.rendering)),
                code_index=deps.code_index,
            )
        if (
            isinstance(root, DatasetRoot)
            and deps.snapshot_reader.read_version(root.spec_id, root.content_version) is None
        ):
            raise RootNotStored(
                spec_id=root.spec_id,
                content_version=root.content_version,
                reason="no such version is stored",
            )
        if isinstance(root, SpecRoot) or latest:
            return self._newest(root, path_words=path_words, codes=codes, trace_id=trace_id)
        return self._at_version(root, path_words=path_words, codes=codes, trace_id=trace_id)

    def _at_version(
        self,
        root: DatasetRoot,
        *,
        path_words: frozenset[str] | None,
        trace_id: str | None,
        revoked_under: Sequence[str] = (),
        codes: Mapping[str, Sequence[str]] | None = None,
    ) -> ResolvedRoot:
        """The version `root` names, read in the rendering it names, summed as the planner
        implies (`_implied`) from the version's trace. A sum the door refuses reads the
        version unsummed. `revoked_under` is `load_stored`'s.

        A sum, implied or stated, over a version a party-totals cut minted reads the cut's
        base version with the cut and the sum in one rendering (`_cut_base`), the rendering
        that version's own frame would give, so the sum does not need the cut version's
        file."""
        spec_id, content_version = root.spec_id, root.content_version
        stored = load_stored(
            subject_id=spec_id,
            content_version=content_version,
            deps=self._deps,
            include_artifacts=False,
            revoked_under=revoked_under,
        )
        options = record_options(mint_options(root.rendering), coverage=stored.coverage)
        groupby = (
            None
            if path_words is None
            else self._implied(
                spec_id, content_version, stored.trace, path_words=path_words, codes=codes
            )
        )
        if (groupby is not None or "groupby" in options) and (
            base := self._cut_base(spec_id, content_version)
        ) is not None:
            rendering = root.rendering.model_copy(update={"without_party_totals": True})
            return self._at_version(
                DatasetRoot(spec_id=spec_id, content_version=base, rendering=rendering),
                path_words=path_words,
                trace_id=trace_id,
                revoked_under=revoked_under,
                codes=codes,
            )
        read = partial(self._stored, root, stored, trace_id=trace_id, revoked_under=revoked_under)
        if groupby is None:
            return read(options=options)
        try:
            return read(options={**options, "groupby": groupby})
        except RollupNotSummableError as refused:
            sum_refused = _sum_refused(refused)
            _log_sum_refused(
                spec_id=spec_id, content_version=content_version, by=groupby, refused=sum_refused
            )
            return replace(read(options=options), sum_refused=sum_refused)

    def _cut_base(self, spec_id: str, content_version: str) -> str | None:
        """The base version of `content_version` when a party-totals cut minted it (its record
        names a base and `frame_summed` false), else `None`."""
        version = self._deps.snapshot_reader.read_version(spec_id, content_version)
        if version is None or version.base_content_version is None:
            return None
        return version.base_content_version if version.frame_summed is False else None

    def _implied(
        self,
        spec_id: str,
        content_version: str,
        trace: Mapping[str, Any],
        *,
        path_words: frozenset[str] | None,
        codes: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, Any] | None:
        """The groupby the planner implies for a root of the version `content_version`, whose
        trace is `trace`, read under `path_words` with the selection `codes`
        (`implied_groupby`), or `None` for a root it implies none for."""
        if path_words is None:
            return None
        return implied_groupby(
            self._contributing(spec_id, content_version, trace),
            path_words=path_words,
            code_index=self._deps.code_index,
            codes=codes,
        )

    def _contributing(
        self, spec_id: str, content_version: str, trace: Mapping[str, Any]
    ) -> list[str]:
        """The StatVars with data in the dataset version `content_version`, as bare dcids:
        those its `trace` counts observations for (`summed_refs`), or, for a trace that counts
        none, as a version minted before traces counted them records, the distinct `stat_var`
        cells of the version's stored frame (`_stored_stat_vars`), read once per version."""
        if _counts_recorded(trace):
            return summed_refs(trace)
        key = (spec_id, content_version)
        if key not in self._stat_vars:
            self._stat_vars[key] = self._stored_stat_vars(spec_id, content_version, trace)
        return self._stat_vars[key]

    def _stored_stat_vars(
        self, spec_id: str, content_version: str, trace: Mapping[str, Any]
    ) -> list[str]:
        """The distinct `stat_var` cells of the stored frame of `content_version`, under the
        bounds of the sum that reads the same frame: a frame1 document past its decode bound
        refuses on its recorded size (`refuse_frame1_past_decode_bound`), and a canonical frame
        of more rows than `groupby_fold_max_rows` refuses on its Parquet row count
        (`refuse_past_fold_ceiling`), before any cell is decoded, as the sum's fold would. A
        canonical frame is read for its `stat_var` column alone. A version with no servable
        stored frame, or whose file is gone, names every StatVar its trace resolved: the read
        that needs the frame then refuses as `RootExpired`.

        Raises:
            OverCeilingError: `bytes`, for a frame1 document past its bound, or `rows`, for a
                canonical frame past the fold's.
        """
        resolved = [
            str(entry["ref"]).removeprefix(_DCID) for entry in trace.get("resolved_stat_vars") or ()
        ]
        locations = self._deps.snapshot_reader.read_artifact_locations(spec_id, content_version)
        location = servable_frame_location(locations, now=self._deps.now())
        if location is None:
            return resolved
        refuse_frame1_past_decode_bound(
            location, fold_max_rows=self._deps.settings.groupby_fold_max_rows
        )
        try:
            data = self._deps.artifact_store.get(
                artifact_digest=location.artifact_digest, tier=Tier(location.tier)
            )
        except ArtifactNotFoundError:
            return resolved
        rows = stored_frame_row_count(
            data, media_type=location.media_type, digest=location.artifact_digest
        )
        if rows is not None:
            refuse_past_fold_ceiling(rows, cap=self._deps.settings.groupby_fold_max_rows)
        cells = stored_frame_distinct(
            data,
            media_type=location.media_type,
            digest=location.artifact_digest,
            column=STAT_VAR_COLUMN,
        )
        return sorted({str(cell).removeprefix(_DCID) for cell in cells if cell is not None})

    def _newest(
        self,
        root: SpecRoot | DatasetRoot,
        *,
        path_words: frozenset[str] | None,
        trace_id: str | None,
        codes: Mapping[str, Sequence[str]] | None = None,
    ) -> ResolvedRoot:
        """`root` read at its dataset's newest version, in the rendering it names, summed as
        the planner implies (`_implied`).

        One request reads one version of a dataset. Each root of a dataset is served
        (`_serve_newest`), which rebuilds a stale head and queues a rebuild past the inline
        budget, and a later root keeps the answer only while it reads the version the first
        serve read. Any other root renders from that version (`_at_version`), so a head that
        moves during the request is not read twice."""
        served = served_spec(spec_id=root.spec_id, deps=self._deps, trace_id=trace_id)
        read = self._first_read(served.spec_id)
        newest = self._serve_newest(
            served, root, path_words=path_words, codes=codes, trace_id=trace_id, read=read
        )
        if newest is not None:
            return newest
        # `_serve_newest` records the first read before it returns `None`.
        first = self._first_read(served.spec_id)
        assert first is not None
        version = DatasetRoot(
            spec_id=served.spec_id,
            content_version=first,
            rendering=root.rendering,
        )
        return self._at_version(
            version,
            path_words=path_words,
            codes=codes,
            trace_id=trace_id,
            revoked_under=(root.spec_id,),
        )

    def _serve_newest(
        self,
        served: StoredDefinition,
        root: SpecRoot | DatasetRoot,
        *,
        path_words: frozenset[str] | None,
        trace_id: str | None,
        read: str | None = None,
        codes: Mapping[str, Sequence[str]] | None = None,
    ) -> ResolvedRoot | None:
        """A root of `served` read at its newest version, served in the one rendering it
        reads: summed as the planner implies from the stored head's version and trace
        (`_head_version`), so no rendering is made only to read a trace; a revoked head gives
        no guess and defers to the serve. The first root records the dataset version its serve
        read (`_newest_versions`), and `read` is that version for a later root. `None` when the
        version served implies another sum than the head did, as when the serve rebuilt the
        head or there was none, or when a later root's serve read another dataset version than
        `read`, even into the same derived version: the root is then read from `read`."""
        # The newest data is read in the rendering the root names, as a fresh root of
        # that rendering reads it, so a version recording two renderings stays readable.
        requested = mint_options(root.rendering)
        head = self._head_version(served.spec_id, posted=root.spec_id, path_words=path_words)
        groupby = (
            None
            if head is None
            else self._implied(served.spec_id, *head, path_words=path_words, codes=codes)
        )
        served_manifest: tuple[Manifest, str | None] | None = None
        sum_refused: dict[str, str] | None = None
        operation_id: str | None = None
        if groupby is not None:
            try:
                served_manifest = self._serve(
                    served,
                    MintOptions.model_validate({**requested, "groupby": groupby}),
                    posted=root.spec_id,
                    trace_id=trace_id,
                )
                requested = {**requested, "groupby": groupby}
            except RollupNotSummableError as refused:
                sum_refused = _sum_refused(refused)
            except RefreshRebuildFailed as failed:
                # A rebuild past the inline budget meets the same refusal on the worker.
                if failed.refusal_code != RollupNotSummableError.code:
                    raise
                sum_refused = _rebuild_sum_refused(failed)
                operation_id = failed.operation_id
        if served_manifest is None:
            served_manifest = self._serve(
                served,
                MintOptions.model_validate(requested),
                posted=root.spec_id,
                trace_id=trace_id,
            )
        manifest, base = served_manifest
        if read is None:
            self._newest_versions[served.spec_id] = (manifest.content_version, base)
        elif (base or self._base_version(served.spec_id, manifest.content_version)) != read:
            return None
        if groupby is not None and sum_refused is not None:
            _log_sum_refused(
                spec_id=served.spec_id,
                content_version=manifest.content_version,
                by=groupby,
                refused=sum_refused,
                operation_id=operation_id,
            )
        elif (
            path_words is not None
            and self._implied(
                served.spec_id,
                base or self._base_version(served.spec_id, manifest.content_version),
                manifest.trace,
                path_words=path_words,
                codes=codes,
            )
            != groupby
        ):
            return None
        # The mint resolved a default enrichment against the version it served.
        options = record_options(requested, coverage=manifest.coverage)
        resolved = self._resolved(
            spec_id=served.spec_id,
            content_version=manifest.content_version,
            options=options,
            manifest=manifest,
            base=base,
        )
        return replace(resolved, sum_refused=sum_refused)

    def _head_version(
        self, spec_id: str, *, posted: str, path_words: frozenset[str] | None
    ) -> tuple[str, Mapping[str, Any]] | None:
        """The version the head of `spec_id` names and its trace, read from its record with no
        rendering, or `None` for a root the planner implies no sum for, a subject with no head,
        or a head whose version is revoked. The trace only guesses the sum the serve reads; a
        revoked head gives no guess, since the serve's freshness check may replace it, and the
        version the serve reads is then judged, revocation included, as with no head."""
        if path_words is None:
            return None
        head = self._deps.head_store.read_head(spec_id)
        if head is None:
            return None
        try:
            stored = load_stored(
                subject_id=spec_id,
                content_version=head.content_version,
                deps=self._deps,
                include_artifacts=False,
                revoked_under=(posted,),
            )
        except SnapshotRevoked:
            return None
        return head.content_version, stored.trace

    def _first_read(self, spec_id: str) -> str | None:
        """The dataset version the first newest root of `spec_id` in this request read
        (`_newest_versions`), or `None` before one has."""
        first = self._newest_versions.get(spec_id)
        if first is None:
            return None
        content_version, read = first
        return read or self._base_version(spec_id, content_version)

    def _base_version(self, spec_id: str, content_version: str) -> str:
        """The dataset version `content_version` renders: the base its record keeps for a
        derived version."""
        version = self._deps.snapshot_reader.read_version(spec_id, content_version)
        if version is None or version.base_content_version is None:
            return content_version
        return version.base_content_version

    def _stored(
        self,
        root: DatasetRoot,
        stored: Manifest,
        *,
        options: Mapping[str, Any],
        trace_id: str | None,
        revoked_under: Sequence[str] = (),
    ) -> ResolvedRoot:
        """The version `root` names, whose manifest is `stored`, read with `options`: its
        stored frame, or the rendering those options make of it."""
        deps = self._deps
        spec_id, content_version = root.spec_id, root.content_version
        manifest = stored
        if options.keys() - _BASE_OPTIONS.keys():
            mint = MintOptions.model_validate(options)
            coverage_snapshot = deps.coverage.snapshot()
            rendered = self._recorded_rendering(
                spec_id, content_version, mint, base=lambda: stored, revoked_under=revoked_under
            ) or head_rendering(
                subject_id=spec_id,
                content_version=content_version,
                deps=deps,
                options=mint,
                trace_id=trace_id,
                coverage_snapshot=coverage_snapshot,
                render=self._render,
                revoked_under=revoked_under,
            )
            if rendered is None and not self._render:
                self._refuse_rendering_past_bound(stored.coverage.rows)
                unstored = render_unstored(
                    subject_id=spec_id,
                    content_version=content_version,
                    options=mint,
                    manifest=stored,
                    deps=deps,
                    coverage_snapshot=coverage_snapshot,
                )
                if unstored is not None:
                    return self._unstored(
                        spec_id=spec_id,
                        content_version=content_version,
                        options=options,
                        unstored=unstored,
                        stored=stored,
                    )
            if rendered is None or _data_artifact(rendered) is None:
                raise RootExpired(
                    spec_id=spec_id,
                    content_version=content_version,
                    reason="its stored frame has expired, so its rendering cannot be made",
                )
            manifest = rendered
        # A groupby sum or a party-totals cut that changed the frame answers its own version,
        # made from the version the root names.
        return self._resolved(
            spec_id=spec_id,
            content_version=manifest.content_version,
            options=options,
            manifest=manifest,
            base=content_version if manifest.content_version != content_version else None,
        )

    def _recorded_rendering(
        self,
        spec_id: str,
        content_version: str,
        options: MintOptions,
        *,
        base: Callable[[], Manifest],
        revoked_under: Sequence[str],
    ) -> Manifest | None:
        """The stored version an earlier read made of the dataset version `content_version`
        with the minting options of `options`, a `groupby` sum or a party-totals cut, for an
        analysis root read in that rendering: a pinned dataset root (`_stored`), or a Spec root
        or a dataset root under `latest` whose serve keeps its head (`_serve`). It is the newest
        version a link of that minting records as rendered from `content_version` under the
        serving code index and render format (`versions.rendering_stamp`,
        `recorded_rendering.mintings_rendered_under`), listing the file of
        the rendering. `None` for a rendering that mints nothing, when no version is recorded
        so, and when the newest one cannot serve: its stored frame file is not servable, it or
        its recorded base is revoked under `revoked_under`, or it links no file of the
        rendering. The caller then renders, and an older recorded version never answers in
        place of the newest.

        `base` reads the dataset version `content_version` with its revocation checked under
        `revoked_under`, so a revoked dataset version answers nothing here even when the
        recorded version's record names another base. Its coverage resolves a default
        enrichment, as the render does (`current.head_rendering`). A request repeated thereby
        reads its stored answer with no render.

        Raises:
            SnapshotRevoked: `base` found the dataset version revoked.
        """
        deps = self._deps
        lister = deps.version_lister
        minting = MintingOptions(
            groupby=options.groupby, without_party_totals=options.without_party_totals
        )
        if lister is None or (minting.groupby is None and not minting.without_party_totals):
            return None
        stamp = rendering_stamp(deps.code_index.stamp.content_hash)
        recorded = next(
            (
                version
                for version in lister.list_head_versions(
                    spec_id, head_content_version=content_version
                )
                if minting
                in mintings_rendered_under(
                    version, head_content_version=content_version, code_index_stamp=stamp
                )
            ),
            None,
        )
        if recorded is None:
            return None
        keys = resolved_rendering_keys(
            options, site=deps.settings.site, enrichment_digest=deps.code_index.enrichment_digest
        )(base())
        now = deps.now()
        with phase("snapshot_state"):
            try:
                stored = read_servable_version(
                    deps.snapshot_reader,
                    subject_id=spec_id,
                    content_version=recorded.content_version,
                    now=now,
                    revoked_under=revoked_under,
                )
            except SnapshotRevoked:
                return None
        if servable_frame_location(stored.servable, now=now) is None:
            return None
        manifest = servable_snapshot_manifest(
            stored,
            id=spec_id,
            content_version=recorded.content_version,
            now=now,
            settings=deps.settings,
            rendering_keys=lambda _: keys,
            enrichment_digest=deps.code_index.enrichment_digest,
        )
        return manifest if _data_artifact(manifest) is not None else None

    def _serve(
        self, served: StoredDefinition, options: MintOptions, *, posted: str, trace_id: str | None
    ) -> tuple[Manifest, str | None]:
        """The served definition's head in the rendering `options` names, honouring revocations
        under `posted`, the id the root names, as a request naming that id does, and the
        dataset version the manifest was read from. The serve judges the head's freshness as
        the definition door does, and a head it keeps answers a minting rendering from the
        version recorded for it (`_recorded_rendering`) before rendering one. The dataset
        version read is the raw version a rebuild in the serve materialised, else the head the
        serve answered from, either of which can differ from the base a recorded version's
        record keeps when two heads make the same version, else `None` for its record's base
        (`_base_version`), read only when a caller needs it.

        Raises:
            AnalysisPending: A rebuild runs past the inline budget.
            RefreshRebuildFailed: The rebuild failed and will not be retried.
        """
        revoked_under = (posted,)
        # The head each manifest `load_head` answered was read from, by its version, and the
        # raw version a rebuild materialised.
        read_from: dict[str, str] = {}
        materialised: list[str] = []

        def load_head(content_version: str, coverage_snapshot: CoverageSnapshot) -> Manifest | None:
            base = partial(
                load_stored,
                subject_id=served.spec_id,
                content_version=content_version,
                deps=self._deps,
                include_artifacts=False,
                revoked_under=revoked_under,
            )
            manifest = self._recorded_rendering(
                served.spec_id, content_version, options, base=base, revoked_under=revoked_under
            ) or head_rendering(
                subject_id=served.spec_id,
                content_version=content_version,
                deps=self._deps,
                options=options,
                trace_id=trace_id,
                coverage_snapshot=coverage_snapshot,
                revoked_under=revoked_under,
            )
            if manifest is not None:
                read_from[manifest.content_version] = content_version
            return manifest

        try:
            manifest = serve_spec(
                stored=served,
                options=options,
                trigger="get",
                deps=self._deps,
                trace_id=trace_id,
                load_head=load_head,
                revoked_under=revoked_under,
                on_materialised=materialised.append,
            )
        except OperationSubmitted as submitted:
            operation = submitted.operation
            if operation.state == "failed":
                raise RefreshRebuildFailed(
                    operation_id=operation.id,
                    refusal_code=operation.refusal_code,
                    refusal=operation.refusal,
                ) from None
            raise _pending(operation.id) from None
        base = materialised[-1] if materialised else read_from.get(manifest.content_version)
        return manifest, base

    def _resolved(
        self,
        *,
        spec_id: str,
        content_version: str,
        options: Mapping[str, Any],
        manifest: Manifest,
        base: str | None = None,
    ) -> ResolvedRoot:
        """The root `manifest`, the version `content_version`, read with `options`. A groupby's
        version names `base`, the dataset version it was made from, else its record's base."""
        artifact = _data_artifact(manifest) if options.get("columns") else None
        root = _root(
            spec_id=spec_id,
            content_version=content_version,
            options=options,
            artifact=artifact,
            rows=manifest.coverage.rows,
            source=None if artifact is None else _FrameLayoutFile(artifact),
        )
        if "groupby" not in options:
            return root
        # A groupby answers a version of its own, which names the dataset version it sums
        # (`_listed_base`).
        version = self._deps.snapshot_reader.read_version(spec_id, content_version)
        base = (
            base
            or (version.base_content_version if version is not None else None)
            or content_version
        )
        root = replace(root, base_content_version=_listed_base(options, base))
        rendered = _rendered_count(manifest)
        if rendered is not None:
            # A stored warning is a collapse's from before re-keys, so it summed.
            count = summed_count(version, rendered=rendered, contributors=0)
        else:
            contributors = self._selected(
                self._contributing(spec_id, base, manifest.trace), options["groupby"]
            )
            count = summed_count(
                version,
                rendered=None,
                contributors=len(contributors),
                adds=self._adds(contributors, options["groupby"]),
            )
        return _with_summed(
            root, count, self._selected(summed_refs(manifest.trace), options["groupby"])
        )

    def _selected(self, contributors: Sequence[str], groupby: Mapping[str, Any]) -> list[str]:
        """`contributors` that `groupby` sums: those its `codes` selection keeps after its
        `levels` roll-up (`selected_contributors`), every one with no selection."""
        return selected_contributors(
            contributors,
            index=self._deps.code_index,
            codes=groupby.get("codes"),
            levels=groupby.get("levels"),
        )

    def _adds(self, contributors: Sequence[str], groupby: Mapping[str, Any]) -> bool:
        """Whether `groupby` adds `contributors`, bare dcids, together (`adds_stat_vars`), on
        the summable axes its `by` keeps. A stacked composite's component sum is not told
        apart here; its components share their kept codes, so they add."""
        index = self._deps.code_index
        by = groupby.get("by") or ()
        kept = sorted(summable_axes(index.declarations) & frozenset(by))
        return adds_stat_vars(
            contributors,
            index=index,
            kept=kept,
            levels=dict(groupby.get("levels") or {}),
            components=False,
        )

    def _unstored(
        self,
        *,
        spec_id: str,
        content_version: str,
        options: Mapping[str, Any],
        unstored: UnstoredRendering,
        stored: Manifest,
    ) -> ResolvedRoot:
        """The root `unstored`, a rendering of the version `content_version` whose manifest is
        `stored`, renders: read from its frame-layout bytes when the root names `columns`, else
        from the derived version's canonical frame it made, else, when the rendering answers
        the version read, from that version's stored frame (`locate`). A groupby names the
        StatVars it summed as a stored rendering does (`stat_vars_summed`)."""
        columns = bool(options.get("columns"))
        source: _Handle | None
        if columns:
            source = _UnstoredFrameLayout(unstored.data)
        elif unstored.frame is not None:
            source = _UnstoredFrame(unstored.frame)
        else:
            source = None
        root = _root(
            spec_id=spec_id,
            content_version=unstored.content_version,
            options=options,
            artifact=unstored.artifact if columns else None,
            rows=unstored.rows,
            source=source,
            base_content_version=_listed_base(options, content_version),
        )
        if "groupby" not in options or unstored.content_version == content_version:
            return root
        contributors = self._selected(
            self._contributing(spec_id, content_version, stored.trace), options["groupby"]
        )
        count = summed_count(
            None,
            rendered=None,
            contributors=len(contributors),
            adds=self._adds(contributors, options["groupby"]),
        )
        return _with_summed(
            root, count, self._selected(summed_refs(stored.trace), options["groupby"])
        )

    def locate(self, root: ResolvedRoot) -> LocatedRoot:
        """`root`, with the version's servable stored frame located when `read` reads it, and a
        frame1 document past `render.frame1_decode_max_bytes` refused on its recorded size. A
        root read from its frame-layout file returns as it is.

        Raises:
            RootExpired: The version has no servable stored frame.
            OverCeilingError: `bytes`, for a frame1 document past the bound.
        """
        located = self._located(root)
        self._located_rows += located.rows or 0
        return located

    def _located(self, root: ResolvedRoot) -> LocatedRoot:
        if isinstance(root.source, _Handle):
            return LocatedRoot.of(root)
        spec_id, content_version = root.dataset.spec_id, root.dataset.content_version
        locations = self._deps.snapshot_reader.read_artifact_locations(spec_id, content_version)
        location = servable_frame_location(locations, now=self._deps.now())
        if location is None:
            raise RootExpired(
                spec_id=spec_id,
                content_version=content_version,
                reason="its stored frame has expired",
            )
        refuse_frame1_past_decode_bound(
            location, fold_max_rows=self._deps.settings.groupby_fold_max_rows
        )
        return LocatedRoot.of(replace(root, source=_LocatedFrame(location)))

    def read(self, root: LocatedRoot) -> Frame:
        """The frame of `root` as `locate` returned it.

        Raises:
            RootExpired: Its file or stored frame is gone.
            TypeError: `root` is not located.
        """
        handle = root.source
        if not isinstance(handle, _Handle):
            raise TypeError(f"a root is read only once `locate` locates it, not from {handle!r}")
        if isinstance(handle, _UnstoredFrameLayout):
            return frame_from_frame_layout(handle.data)
        if isinstance(handle, _UnstoredFrame):
            file = handle.file
            return read_stored_frame(file.data, media_type=file.media_type, digest=file.digest)
        store = self._deps.artifact_store
        spec_id, content_version = root.dataset.spec_id, root.dataset.content_version
        if isinstance(handle, _FrameLayoutFile):
            try:
                data = store.get(
                    artifact_digest=handle.artifact.artifact_digest, tier=Tier(handle.artifact.tier)
                )
            except ArtifactNotFoundError:
                raise RootExpired(
                    spec_id=spec_id, content_version=content_version, reason="its file is gone"
                ) from None
            return frame_from_frame_layout(data)
        location = handle.location
        try:
            data = store.get(artifact_digest=location.artifact_digest, tier=Tier(location.tier))
        except ArtifactNotFoundError:
            raise RootExpired(
                spec_id=spec_id, content_version=content_version, reason="its stored frame is gone"
            ) from None
        return read_stored_frame(
            data, media_type=location.media_type, digest=location.artifact_digest
        )

    def recipe(self, *, spec_id: str, content_version: str) -> Mapping[str, Any] | None:
        record = self._recipes.read_recipe(spec_id=spec_id, content_version=content_version)
        return None if record is None else record.body

    def dataset_root(self, record: Mapping[str, Any]) -> DatasetRoot:
        return recorded_dataset_root(record)


def newest_spec_version(
    spec_id: str, *, deps: DefinitionDeps, recipes: RecipeReader, trace_id: str | None
) -> FrameRef:
    """The dataset version a Spec root `{spec_id}` reads, in its default rendering, named by the
    served id it is recorded under. A bare `spec_id` read by the Presentation frames door, a
    component create or an export resolves through this.

    Raises:
        AnalysisPending: The version is building.
        Exception: Any refusal of the serve, as `LiveRootSource.resolve` raises it.
    """
    root = LiveRootSource(deps=deps, recipes=recipes).resolve(
        SpecRoot(spec_id=spec_id), latest=False, trace_id=trace_id
    )
    return root.dataset


def _pending(operation_id: str) -> AnalysisPending:
    return AnalysisPending(
        operations=(operation_id,), retry_after_s=OPERATION_SUBMIT_RETRY_AFTER_MS / 1000
    )


__all__ = [
    "KIND_COLUMNS",
    "LiveRootSource",
    "frame_from_frame_layout",
    "frame_ref_for",
    "mint_options",
    "newest_spec_version",
    "record_options",
    "requests_kinds",
]
