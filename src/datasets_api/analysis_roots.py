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
- With no `columns` the frame is the dataset version's stored frame, its canonical frame (or
  frame1 document).
- With `columns` the frame is the dataset version's frame-layout file of those columns, decoded
  as data-mcp decodes it: every column outside the grammar's reserved set is a role key, an
  empty enrichment cell reads `untyped`, and a text `unit` cell holds a unit ref or the JCS text
  of a converted unit.

The frame is named by the pair `frame_ref_for` derives from the columns, the groupby and the
party-totals cut, as data-mcp names it, so two renderings of one version are two frames.

A root that leaves `columns` to the default is read with the version's default enrichment
(`default_enrichment`): the root asks the mint for `enrichment: "default"`, which resolves the
columns from the roles the version's frame holds, and its record keeps the resolved `columns`
(`record_options`). The verbs read them to keep group totals out of a sum or rank and to rank
places at one level.

A request root that states no `groupby` and whose version's frame holds more than one StatVar
is read summed (`implied_groupby`), unless a step over it names `stat_var`: its options gain
`groupby` `{by: <the summable axes its steps name, else ["time_period"]>, op: "sum"}`, so its
record, the recipe's normal form and the format 2 ref state the sum. When the sum refuses
`rollup_not_summable` the root is read unsummed and records the refusal (`ResolvedRoot.sum_refused`),
and so does a root read at its newest version whose summed rebuild ran as a queued operation that
failed `rollup_not_summable`. A root whose groupby summed StatVars names them
(`ResolvedRoot.stat_vars_summed`).

A Spec root is served as the definition door serves a registered id (`doors.definition`): its
newest version of the rendering, rebuilt when its head is stale, in one serve. A dataset root under `latest`
is served the same way with the rendering it records, so it is read by its rendering, since its
version alone can record two renderings that the newest data splits. A rebuild
over the inline budget answers `AnalysisPending` naming its operation, and a rebuild that failed
and will not be retried refuses as the refresh door refuses it (`RefreshRebuildFailed`).
`newest_spec_version` names the dataset version a Spec root reads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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
    STATUS_COLUMN,
    UNIT_COLUMN,
    UnitRequest,
    unit_cell_from_json,
)

from artifact_store import Tier
from artifact_store.errors import ArtifactNotFoundError
from code_index.query import CodeIndex
from code_index.types import summable_axes
from dataset_store.snapshot import StoredArtifactLocation
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
from datasets_api.manifest_models import ArtifactEntryModel, CoverageModel, Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.operations.submit import OPERATION_SUBMIT_RETRY_AFTER_MS, OperationSubmitted
from datasets_api.recipe_store import RecipeReader
from datasets_api.render import (
    is_statistics_rendering,
    refuse_frame1_past_decode_bound,
    servable_frame_location,
)
from datasets_api.store_errors import read_stored_frame
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.place_type import (
    UNTYPED,
    enrichment_role,
)
from materialise.reshape import RollupNotSummableError
from spec_registry.reader import StoredDefinition

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
    true` when the cut is asked for, and the `enrichment_stamp` of its file when the columns
    ask for kinds. data-mcp's `frame_materialisation.frame_ref_for` names frames the same way.

    The pair names what a recipe replays: a cut that leaves nothing out serves its base
    version, and the frame read with it is still named apart from the frame read without it,
    since the two replay differently once the data holds party totals."""
    if not columns and groupby is None and not without_party_totals:
        return FrameRef(spec_id=spec_id, content_version=content_version)
    preimage: dict[str, Any] = {"columns": sorted(columns), "content_version": content_version}
    if groupby is not None:
        preimage["groupby"] = {**groupby, "by": sorted(groupby["by"])}
    if without_party_totals:
        preimage["without_party_totals"] = True
    if enrichment_stamp is not None and requests_kinds(columns):
        preimage["enrichment_stamp"] = enrichment_stamp
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


def _contributing_stat_vars(trace: Mapping[str, Any]) -> list[str]:
    """The bare dcids of the StatVars a version's `trace` resolved and its frame holds: each
    `resolved_stat_vars` entry but one recording no observations."""
    entries = trace.get("resolved_stat_vars") or ()
    return [
        str(entry["ref"]).removeprefix("dcid:")
        for entry in entries
        if entry.get("observation_count") != 0
    ]


def implied_groupby(
    trace: Mapping[str, Any], *, path_words: frozenset[str], code_index: CodeIndex
) -> dict[str, Any] | None:
    """The `groupby` a root read with no stated one is read with, from its version's `trace`
    and the words the steps over it name: none when the words name `stat_var` or the frame
    holds one StatVar, else `{by, op: "sum"}`, `by` the summable axes the version's StatVars
    decode (`code_index` declarations) that the words name, as the axis or as its
    `stat_var_<axis>` column, else `["time_period"]`. The datasets door sums every other
    summable axis away, and `stat_var` with them (conformance C120)."""
    if STAT_VAR_WORD in path_words:
        return None
    stat_vars = _contributing_stat_vars(trace)
    if len(stat_vars) < 2:
        return None
    summable = summable_axes(code_index.declarations)
    axes: set[str] = set()
    for dcid in stat_vars:
        if (row := code_index.statvar_by_dcid(dcid)) is not None:
            axes.update(summable.intersection(row.constraint_properties))
    named = {word.removeprefix(STAT_VAR_PREFIX) for word in path_words} & axes
    return {"by": sorted(named) or ["time_period"], "op": "sum"}


def _sum_refused(refused: RollupNotSummableError) -> dict[str, str]:
    return {"axis": refused.axis, "precondition": refused.precondition}


# The precondition a queued rebuild's `rollup_not_summable` names when its operation records
# no refusal body to read it from.
_UNRECORDED = "unrecorded"


def _rebuild_sum_refused(failed: RefreshRebuildFailed) -> dict[str, str]:
    """The `{axis, precondition}` of the `rollup_not_summable` a queued rebuild met: the axis of
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
) -> ResolvedRoot:
    """The root read at `{spec_id, content_version}` with `options`, its frame named by
    `frame_ref_for` and `artifact`'s enrichment stamp, read through `source`."""
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
    ) -> ResolvedRoot:
        deps = self._deps
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
            return self._newest(root, path_words=path_words, trace_id=trace_id)
        spec_id, content_version = root.spec_id, root.content_version
        stored = load_stored(
            subject_id=spec_id, content_version=content_version, deps=deps, include_artifacts=False
        )
        options = record_options(mint_options(root.rendering), coverage=stored.coverage)
        groupby = self._implied(stored, path_words=path_words)
        if groupby is None:
            return self._stored(root, stored, options=options, trace_id=trace_id)
        try:
            return self._stored(
                root, stored, options={**options, "groupby": groupby}, trace_id=trace_id
            )
        except RollupNotSummableError as refused:
            unsummed = self._stored(root, stored, options=options, trace_id=trace_id)
            return replace(unsummed, sum_refused=_sum_refused(refused))

    def _implied(
        self, manifest: Manifest, *, path_words: frozenset[str] | None
    ) -> dict[str, Any] | None:
        """The groupby the planner implies for a root of `manifest`'s version read under
        `path_words` (`implied_groupby`), or `None` for a root it implies none for."""
        if path_words is None:
            return None
        return implied_groupby(
            manifest.trace, path_words=path_words, code_index=self._deps.code_index
        )

    def _newest(
        self,
        root: SpecRoot | DatasetRoot,
        *,
        path_words: frozenset[str] | None,
        trace_id: str | None,
    ) -> ResolvedRoot:
        """`root` read at its dataset's newest version, in the rendering it names, summed as
        the planner implies (`_implied`)."""
        # The newest data is read in the rendering the root names, as a fresh root of
        # that rendering reads it, so a version recording two renderings stays readable.
        served = served_spec(spec_id=root.spec_id, deps=self._deps, trace_id=trace_id)
        requested = mint_options(root.rendering)
        manifest = self._serve(
            served, MintOptions.model_validate(requested), posted=root.spec_id, trace_id=trace_id
        )
        sum_refused: dict[str, str] | None = None
        if (groupby := self._implied(manifest, path_words=path_words)) is not None:
            summed = {**requested, "groupby": groupby}
            try:
                manifest = self._serve(
                    served,
                    MintOptions.model_validate(summed),
                    posted=root.spec_id,
                    trace_id=trace_id,
                )
                requested = summed
            except RollupNotSummableError as refused:
                sum_refused = _sum_refused(refused)
            except RefreshRebuildFailed as failed:
                # A rebuild past the inline budget meets the same refusal on the worker.
                if failed.refusal_code != RollupNotSummableError.code:
                    raise
                sum_refused = _rebuild_sum_refused(failed)
        # The mint resolved a default enrichment against the version it served.
        options = record_options(requested, coverage=manifest.coverage)
        resolved = self._resolved(
            spec_id=served.spec_id,
            content_version=manifest.content_version,
            options=options,
            manifest=manifest,
        )
        return replace(resolved, sum_refused=sum_refused)

    def _stored(
        self,
        root: DatasetRoot,
        stored: Manifest,
        *,
        options: Mapping[str, Any],
        trace_id: str | None,
    ) -> ResolvedRoot:
        """The version `root` names, whose manifest is `stored`, read with `options`: its
        stored frame, or the rendering those options make of it."""
        deps = self._deps
        spec_id, content_version = root.spec_id, root.content_version
        manifest = stored
        if options.keys() - _BASE_OPTIONS.keys():
            mint = MintOptions.model_validate(options)
            coverage_snapshot = deps.coverage.snapshot()
            rendered = head_rendering(
                subject_id=spec_id,
                content_version=content_version,
                deps=deps,
                options=mint,
                trace_id=trace_id,
                coverage_snapshot=coverage_snapshot,
                render=self._render,
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
                    return self._unstored(spec_id=spec_id, options=options, unstored=unstored)
            if rendered is None or _data_artifact(rendered) is None:
                raise RootExpired(
                    spec_id=spec_id,
                    content_version=content_version,
                    reason="its stored frame has expired, so its rendering cannot be made",
                )
            manifest = rendered
        # A groupby sum or a party-totals cut that changed the frame answers its own version.
        return self._resolved(
            spec_id=spec_id,
            content_version=manifest.content_version,
            options=options,
            manifest=manifest,
        )

    def _serve(
        self, served: StoredDefinition, options: MintOptions, *, posted: str, trace_id: str | None
    ) -> Manifest:
        """The served definition's head in the rendering `options` names, honouring revocations
        under `posted`, the id the root names, as a request naming that id does.

        Raises:
            AnalysisPending: A rebuild runs past the inline budget.
            RefreshRebuildFailed: The rebuild failed and will not be retried.
        """
        try:
            return serve_spec(
                stored=served,
                options=options,
                trigger="get",
                deps=self._deps,
                trace_id=trace_id,
                revoked_under=(posted,),
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

    def _resolved(
        self, *, spec_id: str, content_version: str, options: Mapping[str, Any], manifest: Manifest
    ) -> ResolvedRoot:
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
        return replace(root, stat_vars_summed=self._summed(spec_id, content_version, manifest))

    def _summed(self, spec_id: str, content_version: str, manifest: Manifest) -> tuple[str, ...]:
        """The StatVars the groupby collapse `{spec_id, content_version}` summed, as its
        description record's `sum_variables` step names them (`summed_refs`), or none for a
        version that is no collapse of several."""
        version = self._deps.snapshot_reader.read_version(spec_id, content_version)
        if version is None or not version.frame_summed:
            return ()
        summed = tuple(summed_refs(manifest.trace))
        return summed if len(summed) > 1 else ()

    @staticmethod
    def _unstored(
        *, spec_id: str, options: Mapping[str, Any], unstored: UnstoredRendering
    ) -> ResolvedRoot:
        """The root `unstored` renders: read from its frame-layout bytes when the root names
        `columns`, else from the derived version's canonical frame it made, else, when the
        rendering answers the version read, from that version's stored frame (`locate`)."""
        columns = bool(options.get("columns"))
        source: _Handle | None
        if columns:
            source = _UnstoredFrameLayout(unstored.data)
        elif unstored.frame is not None:
            source = _UnstoredFrame(unstored.frame)
        else:
            source = None
        return _root(
            spec_id=spec_id,
            content_version=unstored.content_version,
            options=options,
            artifact=unstored.artifact if columns else None,
            rows=unstored.rows,
            source=source,
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
