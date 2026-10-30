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
from dataclasses import dataclass
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
from dataset_store.snapshot import select_servable_locations
from datasets_api.analysis import (
    AnalysisPending,
    DatasetRoot,
    ResolvedRoot,
    RootExpired,
    RootNotStored,
    RootRendering,
    SpecRoot,
    recorded_dataset_root,
)
from datasets_api.current import DefinitionDeps, head_rendering, load_stored
from datasets_api.default_enrichment import (
    KIND_COLUMNS,
    coverage_roles,
    with_default_enrichment,
)
from datasets_api.derived import root_rendering
from datasets_api.doors.definition import serve_spec, served_spec
from datasets_api.doors.refresh import RefreshRebuildFailed
from datasets_api.manifest_models import ArtifactEntryModel, CoverageModel, Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.operations.submit import OPERATION_SUBMIT_RETRY_AFTER_MS, OperationSubmitted
from datasets_api.recipe_store import RecipeReader
from datasets_api.render import is_statistics_rendering, stored_frame_location
from dfo_analysis_engine.frame_ref import FrameRef
from dfo_analysis_engine.place_type import (
    UNTYPED,
    enrichment_role,
)
from materialise.formats import (
    FRAME1_MEDIA_TYPE,
    frame_entity_columns,
    read_canonical_frame,
    read_frame1_table,
)
from materialise.frame_build import build_frame, unit_cell_for_rebuild
from spec_registry.reader import StoredDefinition

# The bare frame-layout Parquet data-mcp mints a frame with.
_BASE_OPTIONS: Mapping[str, Any] = {"format": "parquet", "bundle": False, "layout": "frame"}
_RESERVED_COLUMNS = frozenset(
    {"stat_var", "time_period", "value", "unit", "measurementMethod", "provenance", "status"}
)
_OPTIONAL_COLUMNS = frozenset({UNIT_COLUMN, "measurementMethod", "provenance"})
_UNIT_TYPE = canonical_frame_schema((UNIT_COLUMN,)).field(UNIT_COLUMN).type


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


@dataclass(frozen=True, kw_only=True)
class _Handle:
    """What `read` reads: the frame-layout file, or the version's stored frame."""

    artifact: ArtifactEntryModel | None


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


def _frame_of_frame1(data: bytes) -> Frame:
    table = read_frame1_table(data)
    return build_frame(
        role_keys=frame_entity_columns(table.columns),
        rows=[
            {
                column: unit_cell_for_rebuild(cell)
                for column, cell in zip(table.columns, row, strict=True)
                if column != STATUS_COLUMN
            }
            for row in table.rows
        ],
    )


def frame_of_stored(data: bytes, *, media_type: str) -> Frame:
    """The frame a version's stored frame `data` holds: its frame1 document when `media_type`
    names one, else its canonical frame."""
    if media_type == FRAME1_MEDIA_TYPE:
        return _frame_of_frame1(data)
    return read_canonical_frame(data)


class LiveRootSource:
    """`analysis.RootSource` over the Datasets surface's stores, for one request.

    `deps` is the request's retrieval view (`doors.definition.request_definition_deps`)."""

    def __init__(self, *, deps: DefinitionDeps, recipes: RecipeReader) -> None:
        self._deps = deps
        self._recipes = recipes

    def resolve(
        self, root: SpecRoot | DatasetRoot, *, latest: bool, trace_id: str | None
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
            # The newest data is read in the rendering the root names, as a fresh root of
            # that rendering reads it, so a version recording two renderings stays readable.
            served = served_spec(spec_id=root.spec_id, deps=deps, trace_id=trace_id)
            requested = mint_options(root.rendering)
            manifest = self._serve(
                served,
                MintOptions.model_validate(requested),
                posted=root.spec_id,
                trace_id=trace_id,
            )
            # The mint resolved a default enrichment against the version it served.
            options = record_options(requested, coverage=manifest.coverage)
            return self._resolved(
                spec_id=served.spec_id,
                content_version=manifest.content_version,
                options=options,
                manifest=manifest,
            )
        spec_id, content_version = root.spec_id, root.content_version
        stored = load_stored(
            subject_id=spec_id, content_version=content_version, deps=deps, include_artifacts=False
        )
        options = record_options(mint_options(root.rendering), coverage=stored.coverage)
        manifest = stored
        if options.keys() - _BASE_OPTIONS.keys():
            rendered = head_rendering(
                subject_id=spec_id,
                content_version=content_version,
                deps=deps,
                options=MintOptions.model_validate(options),
                trace_id=trace_id,
                coverage_snapshot=deps.coverage.snapshot(),
            )
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
                    operation_id=operation.id, refusal_code=operation.refusal_code
                ) from None
            raise _pending(operation.id) from None

    def _resolved(
        self, *, spec_id: str, content_version: str, options: Mapping[str, Any], manifest: Manifest
    ) -> ResolvedRoot:
        columns = tuple(options.get("columns") or ())
        artifact = _data_artifact(manifest) if columns else None
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
            rows=manifest.coverage.rows,
            source=_Handle(artifact=artifact),
        )

    def read(self, root: ResolvedRoot) -> Frame:
        handle = root.source
        assert isinstance(handle, _Handle)
        store = self._deps.artifact_store
        spec_id, content_version = root.dataset.spec_id, root.dataset.content_version
        if handle.artifact is not None:
            try:
                data = store.get(
                    artifact_digest=handle.artifact.artifact_digest, tier=Tier(handle.artifact.tier)
                )
            except ArtifactNotFoundError:
                raise RootExpired(
                    spec_id=spec_id, content_version=content_version, reason="its file is gone"
                ) from None
            return frame_from_frame_layout(data)
        locations = self._deps.snapshot_reader.read_artifact_locations(spec_id, content_version)
        location = stored_frame_location(select_servable_locations(locations, now=self._deps.now()))
        if location is None:
            raise RootExpired(
                spec_id=spec_id,
                content_version=content_version,
                reason="its stored frame has expired",
            )
        try:
            data = store.get(artifact_digest=location.artifact_digest, tier=Tier(location.tier))
        except ArtifactNotFoundError:
            raise RootExpired(
                spec_id=spec_id, content_version=content_version, reason="its stored frame is gone"
            ) from None
        return frame_of_stored(data, media_type=location.media_type)

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
    "frame_of_stored",
    "frame_ref_for",
    "mint_options",
    "newest_spec_version",
    "record_options",
    "requests_kinds",
]
