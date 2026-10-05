"""Renders materialised frames as CSV, XLSX, or Parquet, optionally packaged as a Frictionless
Data Package. Decode, groupby, selection, enrichment, and pivot run before the writer. Reshaping
preserves the materialisation's `content_version`.

Decode runs first and adds `stat_var_{code_system}` columns, so selection and pivot can name them.
`options.columns` resolves a token to a real or decoded column first, then to an enabled
`{entity_column}_{field}` enrichment (for example `recipient_iso3`). Table and decoded columns
select; a list that names none keeps every default column and only adds the enrichments. Unknown
tokens list the parseable qualified tokens as `alternatives`.

`groupby` and `pivot` read across chunks. Other reshapes stream rows to the writer. A recognised but
disabled enrichment token raises `capability_unavailable`. Unknown tokens raise `column_unknown`.
Collections store one artifact per member when bare, or one package with a resource per member.

A mint of at most `FRAME1_MAX_ROWS` rows also streams canonical frame1 to the store for
presentation. Frame1 is an unsigned internal rendering on the version record. Series statistics
are stored separately from the requested rendering.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import IO, cast
from urllib.parse import urlencode

from artifact_store import (
    ArtifactRecordConflictError,
    ArtifactSigningError,
    ArtifactStore,
    StoredArtifact,
    Tier,
)
from code_index.labels import family_label
from code_index.members import absent_member_reads_zero
from code_index.query import CodeIndex
from code_index.shared import get_measure_index
from code_index.types import summable_axes
from dataset_store.snapshot import StoredArtifactLocation
from datasets_api.adapter import pinned_member_ref
from datasets_api.artifact_delivery import (
    artifact_content_disposition,
    artifact_file_name,
)
from datasets_api.capabilities import ENABLED_ENRICHMENTS
from datasets_api.codes_materialise import CapabilityUnavailable, RenderedArtifactSigningFailed
from datasets_api.config import SIGNED_URL_ADVERTISED_TTL, SIGNED_URL_TTL
from datasets_api.decode import build_decode_plan
from datasets_api.groupby import (
    collapse_axes,
    contributing_dcids,
    declared_axes_present,
    is_full_collapse,
    refuse_decode_gaps,
    refuse_provenance_disagreement,
    refuse_stat_var_collision,
    refuse_unknown_by_axes,
    unsummed_axes,
)
from datasets_api.manifest_models import ArtifactEntryModel, CitationSource, Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.timings import phase
from datasets_api.versions import ArtifactRecord
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from dfo_platform import Alternative
from dfo_platform.config import PublicSiteSettings
from materialise.bundle import (
    ArtifactEntry as BundleArtifactEntry,
)
from materialise.bundle import (
    BundleContext,
    BundleLinks,
    CodeEntry,
    MemberProvenance,
    ReproducingCall,
    SourceProvenance,
    assemble_bundle,
)
from materialise.bundle_columns import (
    BundleCodes,
    TableScan,
    bundle_codes,
    column_docs,
    summary_line,
    year_window,
)
from materialise.bundle_naming import heading, member_slugs, sources_slug
from materialise.bundle_sources import source_provenance
from materialise.codes import MaterialisationWarning
from materialise.content_version import MintPreimage, mint_version
from materialise.enrichment import EnrichmentLookup
from materialise.formats import (
    FORMAT_COLUMN_CAPS,
    Frame,
    Rendering,
    Table,
    WrittenArtifact,
    table_schema_for,
    write_csv,
    write_excel,
    write_frame1,
    write_parquet,
)
from materialise.frame_build import build_frame, unit_cell_for_rebuild
from materialise.reshape import (
    SUPPORTED_ENRICHMENTS,
    ColumnClassification,
    ColumnUnknownError,
    DecodePlan,
    EnrichmentPlan,
    EnrichmentUnavailableWarning,
    ExpectedPopulation,
    GroupbyPlan,
    GroupbySumChunksOutcome,
    ParsedColumns,
    PivotPlan,
    ReshapePlan,
    RollupNotSummableError,
    apply_reshapes,
    classify_columns,
    enrichment_unavailable_warning,
    fold_groupby_sum_chunks,
    fold_pivot_chunks,
    parse_column_tokens,
    select_columns,
)
from materialise.reshape import Table as ReshapeTable

_ZIP_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)

_FORMAT_WRITERS: Mapping[str, Callable[..., WrittenArtifact]] = {
    "csv": write_csv,
    "xlsx": write_excel,
    "parquet": write_parquet,
}

# The writer map uses `excel` for Excel. The wire option uses `xlsx`.
_FORMAT_COLUMN_CAP_KEYS: Mapping[str, str] = {"csv": "csv", "xlsx": "excel", "parquet": "parquet"}

_STAT_VAR_COLUMN = "stat_var"
"""Shared with decode and groupby. A non-empty collapse set includes `stat_var`."""

_STAT_VAR_REF_PREFIX = "dcid:"
"""Mirrors `datasets_api.groupby`'s own constant of the same name -- the
family ref a full collapse mints when more than one member contributed
rows."""

_STATUS_COLUMN = "status"
"""`Frame.build` reserves `status`. `_full_collapse_frame` strips it before rebuilding. The local
constant avoids importing the grammar package for one name."""


class StatVarsSummedWarning(MaterialisationWarning):
    """`stat_vars_summed`'s own wider warning shape: the standard
    `{code, severity, message}` plus `contributing_member_count`,
    `stat_var_ref`, and `stat_var_label`, the facts `datasets_api.manifest
    .reshape_warning_models` puts on the wire warning and names in the `message` it composes.
    `EnrichmentUnavailableWarning` (`materialise.reshape`) gets the same treatment there.
    `stat_var_label` is `code_index.labels.family_label` over the subject's identity, the label
    the presentation door gives the same family ref, or `None` when it has none.
    """

    contributing_member_count: int
    stat_var_ref: str
    stat_var_label: str | None


def _stat_vars_summed_warning(
    *, contributing_member_count: int, stat_var_ref: str, stat_var_label: str | None
) -> StatVarsSummedWarning:
    _title, detail, _severity = DATASETS_WARNING_CATALOGUE["stat_vars_summed"]
    return {
        "code": "stat_vars_summed",
        "severity": "info",
        "message": detail,
        "contributing_member_count": contributing_member_count,
        "stat_var_ref": stat_var_ref,
        "stat_var_label": stat_var_label,
    }


@dataclass(frozen=True)
class GroupbyCollapseRendering:
    """The canonical frame from a full collapse, with `stat_var` restored on every row. It is
    independent of selected columns, pivot, and format. An absent mint preimage yields `None` for
    `content_version`. `contributing_member_count` counts distinct StatVars with rows.
    Callers use it
    to warn about sums or refuse collection-member collapses."""

    frame: Frame
    content_version: str | None
    stat_var_ref: str
    contributing_member_count: int


@dataclass(frozen=True)
class ReshapeResult:
    """Carries the reshaped table, warnings, and optional full-collapse result. `warning_order`
    records decode-then-enrichment order for merging chunk warnings."""

    table: Table
    warnings: tuple[MaterialisationWarning, ...]
    collapse: GroupbyCollapseRendering | None
    warning_order: tuple[str, ...]


def check_options_supported(options: MintOptions) -> None:
    """Frame-dependent refusals run in `_reshape`. Groupby, decode, and disabled-enrichment checks
    need the materialised frame."""


def _column_cap(format: str) -> int:
    """Returns the format's pivot-width cap as an integer. `None` means uncapped and maps to
    `sys.maxsize`, above any table width."""
    cap = FORMAT_COLUMN_CAPS[_FORMAT_COLUMN_CAP_KEYS[format]]
    return cap if cap is not None else sys.maxsize


def _enrichment_refs(table: Table, *, enrichments: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """The distinct string cell values, first-seen order, that `enrichments`'
    entity columns carry on `table`. This is the ref set `EnrichmentLookup
    .lookup` needs, collected once across every requested entity column,
    since two requests naming the same entity column share one ref set.
    """
    columns = sorted({entity_column for entity_column, _field in enrichments})
    refs: list[str] = []
    seen: set[str] = set()
    for column in columns:
        pos = table.columns.index(column)
        for row in table.rows:
            ref = row[pos]
            if isinstance(ref, str) and ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return tuple(refs)


def _groupby_population(
    *,
    decoded_code_systems: Sequence[str],
    resolved_stat_var_refs: Iterable[str],
    lookup: Mapping[str, Mapping[str, str]],
) -> tuple[tuple[str, ...], frozenset[tuple[object, ...]]]:
    """Projects each resolved StatVar to `(stat_var, decoded values...)` for groupby population
    checks. It uses the same decode lookup as the table. `refuse_decode_gaps` must run first because
    this function indexes every requested code system directly."""
    columns = (_STAT_VAR_COLUMN, *(f"{_STAT_VAR_COLUMN}_{cs}" for cs in decoded_code_systems))
    members = frozenset(
        (ref, *(lookup[ref][cs] for cs in decoded_code_systems)) for ref in resolved_stat_var_refs
    )
    return columns, members


_RESHAPE_CHUNK_ROWS = 8192
"""The 8,192-row reshape cap matches writer and profile slices. `materialise.reshape` avoids the
frame-grammar import, so this door re-slices Arrow frames before row-reading reshapes. A 100,000-row
chunk can otherwise trigger a `to_pylist()` rebuild about 12 times larger than writer chunks."""


def _reslice_chunks(chunks: Iterable[Table]) -> Iterator[Table]:
    """Uses zero-copy slices for Arrow frames and tuple slices for other tables. Empty chunks pass
    through because an empty materialisation can yield one. `_reshape_stream` applies the 8,192-row
    bound before each streaming path."""
    for chunk in chunks:
        if isinstance(chunk, Frame):
            whole = chunk.table
            if whole.num_rows == 0:
                yield chunk
                continue
            for start in range(0, whole.num_rows, _RESHAPE_CHUNK_ROWS):
                yield cast(
                    Table,
                    Frame(columns=chunk.columns, table=whole.slice(start, _RESHAPE_CHUNK_ROWS)),
                )
        else:
            rows = chunk.rows
            if not rows:
                yield chunk
                continue
            for start in range(0, len(rows), _RESHAPE_CHUNK_ROWS):
                yield cast(
                    Table,
                    ReshapeTable(
                        columns=chunk.columns, rows=rows[start : start + _RESHAPE_CHUNK_ROWS]
                    ),
                )


def _stream_first_axis_collapse(
    frame_chunks: Iterable[Table],
    *,
    axis: str,
    classification: ColumnClassification,
    summable_axes: frozenset[str],
    missing_declared_as_zero: bool,
    expected_population: ExpectedPopulation,
    decode_plan: DecodePlan | None,
) -> tuple[GroupbySumChunksOutcome, frozenset[str], tuple[MaterialisationWarning, ...]]:
    """Decodes each chunk and folds it into `axis` in one pass, retaining one chunk plus per-group
    accumulators. Collects distinct StatVars for the caller's collision check. `_reshape` checks
    collisions before raising fold violations, matching whole-table error order."""
    distinct_dcids: set[str] = set()
    stat_var_pos: list[int] = []
    warnings: list[MaterialisationWarning] = []

    def _decoded_chunks() -> Iterator[Table]:
        for chunk in frame_chunks:
            if decode_plan is not None:
                decoded, decode_warnings = apply_reshapes(
                    chunk, plan=ReshapePlan(decode=decode_plan)
                )
                decoded = cast(Table, decoded)
                warnings.extend(decode_warnings)
            else:
                decoded = chunk
            if not stat_var_pos:
                stat_var_pos.append(decoded.columns.index(_STAT_VAR_COLUMN))
            pos = stat_var_pos[0]
            for row in decoded.rows:
                cell = row[pos]
                if isinstance(cell, str):
                    distinct_dcids.add(cell)
            yield decoded

    outcome = fold_groupby_sum_chunks(
        _decoded_chunks(),
        axis=axis,
        classification=classification,
        summable_axes=summable_axes,
        missing_declared_as_zero=missing_declared_as_zero,
        expected_population=expected_population,
    )
    return outcome, frozenset(distinct_dcids), tuple(warnings)


@dataclass(frozen=True)
class _GroupbyOutcome:
    """Carries the groupby table, pending decode plan, warnings, and optional full-collapse
    rendering."""

    table: Table
    decode_plan: DecodePlan | None
    warnings: tuple[MaterialisationWarning, ...]
    collapse: GroupbyCollapseRendering | None


def _reshape_groupby_collapse(
    table: Table,
    current: Table,
    *,
    decode_plan: DecodePlan | None,
    axes_to_collapse: Sequence[str],
    full_collapse: bool,
    decoded_code_systems: Sequence[str],
    code_index: CodeIndex,
    provenance_refs: Sequence[str],
    entity_columns: Sequence[str],
    subject_id: str | None,
    subject_label: str | None,
    groupby_preimage: MintPreimage | None,
    frame_chunks: Iterable[Table] | None,
    warnings: list[MaterialisationWarning],
) -> tuple[Table, DecodePlan | None, GroupbyCollapseRendering | None]:
    """Collapses `stat_var` then each requested axis, restoring `stat_var` to a frame on full
    collapse. That frame leaves out the column of each decoded axis the sum does not collapse
    (`groupby.unsummed_axes`), which a full collapse holds to one value. A member with no rows
    counts as zero when the frame's dataset declares it (`absent_member_reads_zero`). Appends
    decode and streaming warnings to `warnings` and returns any unconsumed decode plan. A full
    collapse requires `subject_id` unless one member contributed. `subject_label` labels the
    family ref on the `stat_vars_summed` warning."""
    refuse_provenance_disagreement(provenance_refs=provenance_refs)
    decode_lookup = decode_plan.lookup if decode_plan is not None else {}
    if frame_chunks is None:
        if decode_plan is not None:
            current, decode_warnings = apply_reshapes(current, plan=ReshapePlan(decode=decode_plan))
            current = cast(Table, current)
            warnings.extend(decode_warnings)
            decode_plan = None  # Already applied. The closing apply_reshapes skips it.
        refuse_decode_gaps(
            lookup=decode_lookup,
            refs=decode_lookup.keys(),
            code_systems=decoded_code_systems,
        )
        refuse_stat_var_collision(
            current, index=code_index, decoded_code_systems=decoded_code_systems
        )
        # `contributing_dcids` counts members with rows. Resolved zero-row members stay out.
        contributors = contributing_dcids(current)
    else:
        # `current` is one representative chunk, so the fold gathers contributors across the stream.
        # Decode-gap checks use only `decode_lookup` and run before the fold.
        refuse_decode_gaps(
            lookup=decode_lookup,
            refs=decode_lookup.keys(),
            code_systems=decoded_code_systems,
        )
    # `refuse_provenance_disagreement` leaves one provenance at most: the dataset the frame
    # was served from, which is the elected one for an index-resolved family.
    dataset = next(iter(set(provenance_refs)), None)
    missing_declared_as_zero = absent_member_reads_zero(code_index, dataset)
    summable_columns = frozenset(
        f"{_STAT_VAR_COLUMN}_{code_system}"
        for code_system in summable_axes(code_index.declarations)
    ) | {_STAT_VAR_COLUMN}
    collapse_columns = (
        _STAT_VAR_COLUMN,
        *(f"{_STAT_VAR_COLUMN}_{code_system}" for code_system in axes_to_collapse),
    )
    population_columns, population_members = _groupby_population(
        decoded_code_systems=decoded_code_systems,
        resolved_stat_var_refs=decode_lookup.keys(),
        lookup=decode_lookup,
    )
    remaining_columns = population_columns
    for axis_index, axis in enumerate(collapse_columns):
        positions = tuple(population_columns.index(c) for c in remaining_columns)
        projected_members = frozenset(
            tuple(member[p] for p in positions) for member in population_members
        )
        axis_population = ExpectedPopulation(columns=remaining_columns, members=projected_members)
        if axis_index == 0 and frame_chunks is not None:
            # Fold the first axis with one output-group accumulator and the previous grain key.
            # Later axes operate on the reduced, one-row-per-group table.
            decoded_column_names_before_collapse = (
                tuple(
                    f"{entity_column}_{code_system}"
                    for entity_column, code_system in decode_plan.requests
                )
                if decode_plan is not None
                else ()
            )
            classification = classify_columns(
                (*table.columns, *decoded_column_names_before_collapse)
            )
            outcome, streaming_contributors, decode_warnings = _stream_first_axis_collapse(
                frame_chunks,
                axis=axis,
                classification=classification,
                summable_axes=summable_columns,
                missing_declared_as_zero=missing_declared_as_zero,
                expected_population=axis_population,
                decode_plan=decode_plan,
            )
            warnings.extend(decode_warnings)
            decode_plan = None  # already applied per chunk above
            # Check all streamed StatVars for collisions before raising the fold's violation, as in
            # the whole-table path.
            distinct_table = ReshapeTable(
                columns=(_STAT_VAR_COLUMN,),
                rows=tuple((dcid,) for dcid in sorted(streaming_contributors)),
            )
            refuse_stat_var_collision(
                distinct_table, index=code_index, decoded_code_systems=decoded_code_systems
            )
            if outcome.violation is not None:
                raise outcome.violation
            assert outcome.table is not None
            current = outcome.table
            contributors = contributing_dcids(distinct_table)
        else:
            current, _ = apply_reshapes(
                current,
                plan=ReshapePlan(
                    groupby=GroupbyPlan(
                        axis=axis,
                        summable_axes=summable_columns,
                        missing_declared_as_zero=missing_declared_as_zero,
                        expected_population=axis_population,
                    )
                ),
            )
            current = cast(Table, current)
        remaining_columns = tuple(c for c in remaining_columns if c != axis)

    if not full_collapse:
        return current, decode_plan, None

    if len(contributors) == 1:
        stat_var_ref = next(iter(contributors))
    else:
        if subject_id is None:
            raise ValueError(
                "a full groupby collapse that is not a sole-contributor no-op "
                "requires subject_id (the family ref falls back to "
                "f'dcid:{subject_id}'); the sole-contributor no-op case is the "
                "only one that does not need it"
            )
        stat_var_ref = f"{_STAT_VAR_REF_PREFIX}{subject_id}"
        # Empty collapses need a family ref. `stat_vars_summed` warnings require multiple contributors.
        if len(contributors) > 1:
            warnings.append(
                _stat_vars_summed_warning(
                    contributing_member_count=len(contributors),
                    stat_var_ref=stat_var_ref,
                    stat_var_label=subject_label,
                )
            )
    # A full collapse leaves each unsummed axis with one value, so its column describes the
    # family and the grammar frame leaves it out.
    left_out = {_STATUS_COLUMN} | {
        f"{_STAT_VAR_COLUMN}_{code_system}"
        for code_system in unsummed_axes(
            decoded_code_systems=decoded_code_systems, declarations=code_index.declarations
        )
    }
    # Unwrap `UnitRequest` cells to the shape `build_frame` accepts.
    rows_as_dicts = [
        {
            **{
                column: unit_cell_for_rebuild(value)
                for column, value in zip(current.columns, row, strict=True)
                if column not in left_out
            },
            _STAT_VAR_COLUMN: stat_var_ref,
        }
        for row in current.rows
    ]
    collapsed_frame = build_frame(role_keys=entity_columns, rows=rows_as_dicts)
    current = cast(Table, collapsed_frame)
    new_content_version = (
        mint_version(
            collapsed_frame,
            graph_revision=groupby_preimage.graph_revision,
            source_revisions=groupby_preimage.source_revisions,
            facets=groupby_preimage.facets,
            conversion=groupby_preimage.conversion,
        )
        if groupby_preimage is not None
        else None
    )
    collapse_info = GroupbyCollapseRendering(
        frame=collapsed_frame,
        content_version=new_content_version,
        stat_var_ref=stat_var_ref,
        contributing_member_count=len(contributors),
    )
    return current, decode_plan, collapse_info


def _reshape_groupby(
    table: Table,
    *,
    decode_plan: DecodePlan | None,
    options: MintOptions,
    code_index: CodeIndex | None,
    entity_columns: Sequence[str],
    provenance_refs: Sequence[str],
    subject_id: str | None,
    subject_label: str | None,
    groupby_preimage: MintPreimage | None,
    frame_chunks: Iterable[Table] | None,
) -> _GroupbyOutcome:
    """Applies `groupby` when options and `code_index` are set. Otherwise inputs pass through.
    Unknown `by` axes fail before frame classification. The collapse set is the frame's decoded,
    declared-summable axes absent from `by`.

    For a collapse, decode includes resolved StatVars with zero rows. It checks decode gaps, StatVar
    collisions, and provenance agreement before summing each axis in order, with `stat_var` first.
    Each sum uses its expected population and the source's absent-observation rule. Naming all
    decoded axes in `by` is a no-op. With no summable axis, omitting `stat_var` with multiple
    contributors
    raises `RollupNotSummableError`.

    `frame_chunks` streams the `stat_var` collapse over all input rows. Later axes use the reduced
    output. The streamed path preserves whole-table refusal order. A full collapse restores
    `stat_var` from the sole contributor or `dcid:{subject_id}`. Multiple contributors add a
    `stat_vars_summed` warning labelled `subject_label`. A full collapse requires `subject_id`
    unless one member contributed.
    With `groupby_preimage`, the collapsed frame gets a version from the base revision, facets,
    source revisions, and conversion. A sole-contributor collapse shares the base version. An absent
    preimage yields `None`.

    Raises `RollupNotSummableError` when a family has multiple contributors and no summable axis,
    or `ValueError` when a full collapse lacks `subject_id`.
    """
    current: Table = table
    warnings: list[MaterialisationWarning] = []
    collapse_info: GroupbyCollapseRendering | None = None

    if options.groupby is None or code_index is None:
        return _GroupbyOutcome(table=current, decode_plan=decode_plan, warnings=(), collapse=None)

    refuse_unknown_by_axes(by=options.groupby.by, declarations=code_index.declarations)
    decoded_code_systems = (
        tuple(code_system for _entity_column, code_system in decode_plan.requests)
        if decode_plan is not None
        else ()
    )
    axes_to_collapse = collapse_axes(
        by=options.groupby.by,
        decoded_code_systems=decoded_code_systems,
        declarations=code_index.declarations,
    )
    full_collapse = is_full_collapse(
        by=options.groupby.by,
        decoded_code_systems=decoded_code_systems,
        declarations=code_index.declarations,
        lookup=decode_plan.lookup if decode_plan is not None else {},
    )
    if axes_to_collapse:
        current, decode_plan, collapse_info = _reshape_groupby_collapse(
            table,
            current,
            decode_plan=decode_plan,
            axes_to_collapse=axes_to_collapse,
            full_collapse=full_collapse,
            decoded_code_systems=decoded_code_systems,
            code_index=code_index,
            provenance_refs=provenance_refs,
            entity_columns=entity_columns,
            subject_id=subject_id,
            subject_label=subject_label,
            groupby_preimage=groupby_preimage,
            frame_chunks=frame_chunks,
            warnings=warnings,
        )
    elif (
        _STAT_VAR_COLUMN not in options.groupby.by
        and not declared_axes_present(
            decoded_code_systems=decoded_code_systems, declarations=code_index.declarations
        )
        and len(contributing_dcids(current)) > 1
    ):
        # A family rollup needs a decoded summable axis. `current` is the raw table here, so
        # `contributing_dcids` counts its StatVars.
        raise RollupNotSummableError(
            f"groupby.by omits {_STAT_VAR_COLUMN!r} but this frame decodes no "
            "declared-summable axis to collapse its contributing StatVars over; "
            "this family's members carry no summable axis, so they cannot be summed",
            axis=_STAT_VAR_COLUMN,
            precondition="no_summable_axis",
        )

    return _GroupbyOutcome(
        table=current,
        decode_plan=decode_plan,
        warnings=tuple(warnings),
        collapse=collapse_info,
    )


def _reshape_select_pivot_enrich(
    current: Table,
    *,
    decode_plan: DecodePlan | None,
    options: MintOptions,
    entity_columns: Sequence[str],
    enrichment: EnrichmentLookup,
    pivot_max_output_cells: int,
    warnings: list[MaterialisationWarning],
    collapse: GroupbyCollapseRendering | None,
) -> ReshapeResult:
    """Applies pending decode, `options.columns`, and `options.pivot`. Real and decoded columns take
    precedence over enrichment tokens. Enabled `{entity}_{field}` tokens use one batched lookup.
    Recognised disabled fields raise `CapabilityUnavailable`, and unknown tokens raise
    `ColumnUnknownError` with alternatives. Decode, selection, then pivot run in that order. A token
    list with no table or decoded column does not select, so every default column stays.

    `warning_order` preserves decode-request then enrichment-request order for merging chunk
    warnings. Raises `ColumnUnknownError` for unknown columns and `CapabilityUnavailable` for
    disabled enrichment fields.
    """
    decoded_column_names = (
        tuple(f"{entity_column}_{field}" for entity_column, field in decode_plan.requests)
        if decode_plan is not None
        else ()
    )
    post_decode_columns = (*current.columns, *decoded_column_names)

    parsed: ParsedColumns | None = None
    if options.columns is not None:
        try:
            parsed = parse_column_tokens(
                options.columns,
                table_columns=post_decode_columns,
                entity_columns=entity_columns,
                supported_fields=SUPPORTED_ENRICHMENTS,
            )
        except ColumnUnknownError as exc:
            exc.pointer = "/options/columns"
            exc.alternatives = [
                Alternative(ref=column, label=column) for column in post_decode_columns
            ] + list(exc.alternatives or [])
            raise
        disabled = sorted(
            {
                field
                for _entity_column, field in parsed.enrichments
                if field not in ENABLED_ENRICHMENTS
            }
        )
        if disabled:
            raise CapabilityUnavailable(
                detail=f"enrichment field(s) {disabled!r} are recognised but not enabled on "
                "this deployment; GET /api/datasets/capabilities names the enrichment fields "
                "that are live",
                pointer="/options/columns",
            )

    pivot = (
        PivotPlan(
            index=options.pivot.index,
            columns=options.pivot.columns,
            column_cap=_column_cap(options.format),
            cell_cap=pivot_max_output_cells,
        )
        if options.pivot is not None
        else None
    )

    select = (parsed.select or None) if parsed is not None else None
    enrichment_plan: EnrichmentPlan | None = None
    if parsed is not None and parsed.enrichments:
        refs = _enrichment_refs(current, enrichments=parsed.enrichments)
        enrichment_plan = EnrichmentPlan(
            requests=parsed.enrichments, lookup=enrichment.lookup(refs)
        )
    warning_order = (
        *decoded_column_names,
        *(
            (f"{entity_column}_{field}" for entity_column, field in enrichment_plan.requests)
            if enrichment_plan is not None
            else ()
        ),
    )

    if decode_plan is None and select is None and pivot is None and enrichment_plan is None:
        return ReshapeResult(
            table=current,
            warnings=tuple(warnings),
            collapse=collapse,
            warning_order=warning_order,
        )
    try:
        reshaped, more_warnings = apply_reshapes(
            current,
            plan=ReshapePlan(
                decode=decode_plan, select=select, enrichment=enrichment_plan, pivot=pivot
            ),
        )
    except ColumnUnknownError as exc:
        exc.pointer = "/options/pivot"
        raise
    warnings.extend(more_warnings)
    # `reshape.Table` satisfies `formats.Table` structurally (the same
    # `.columns`/`.rows` shape `Frame`/`materialise.ConcatenatedFrameChunks`
    # both do) but not per strict Protocol variance, since `formats.Table`
    # declares its members as plain (writable) attributes rather than
    # read-only properties, hence the `cast(Table, ...)` below.
    return ReshapeResult(
        table=cast(Table, reshaped),
        warnings=tuple(warnings),
        collapse=collapse,
        warning_order=warning_order,
    )


def _reshape(
    table: Table,
    *,
    options: MintOptions,
    entity_columns: Sequence[str],
    enrichment: EnrichmentLookup,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None = None,
    resolved_stat_var_refs: Sequence[str],
    provenance_refs: Sequence[str],
    subject_id: str | None = None,
    subject_label: str | None = None,
    groupby_preimage: MintPreimage | None = None,
    frame_chunks: Iterable[Table] | None = None,
) -> ReshapeResult:
    """Applies decode, groupby, selection, enrichment, and pivot before rendering. Inert transforms
    leave the input table unchanged.

    A supplied `code_index` enables decode-plan construction even when no decode option is set.
    `None` skips decoding.
    `resolved_stat_var_refs` supplies zero-row members to groupby population checks, and
    `provenance_refs` govern its source-wide missing-as-zero rule. `subject_id` and
    `groupby_preimage` are used only for full-collapse identity, and `subject_label` only for the
    `stat_vars_summed` warning's family label. `frame_chunks` streams the first
    collapse axis. `None` processes `table` in memory.

    Raises `ColumnUnknownError` for unknown columns, `CapabilityUnavailable` for disabled
    enrichment, `RollupNotSummableError` when a family has no summable axis, and `ValueError` when
    a full collapse lacks `subject_id`.
    """
    decode_plan = (
        build_decode_plan(table, index=code_index, extra_refs=resolved_stat_var_refs)
        if code_index is not None
        else None
    )
    groupby_outcome = _reshape_groupby(
        table,
        decode_plan=decode_plan,
        options=options,
        code_index=code_index,
        entity_columns=entity_columns,
        provenance_refs=provenance_refs,
        subject_id=subject_id,
        subject_label=subject_label,
        groupby_preimage=groupby_preimage,
        frame_chunks=frame_chunks,
    )
    return _reshape_select_pivot_enrich(
        groupby_outcome.table,
        decode_plan=groupby_outcome.decode_plan,
        options=options,
        entity_columns=entity_columns,
        enrichment=enrichment,
        pivot_max_output_cells=pivot_max_output_cells,
        warnings=list(groupby_outcome.warnings),
        collapse=groupby_outcome.collapse,
    )


def _sources(
    provenance_refs: Sequence[str], citations: Mapping[str, CitationSource]
) -> tuple[SourceProvenance, ...]:
    """The distinct sources of `provenance_refs`, in first-seen order."""
    return tuple(
        source_provenance(
            ref,
            cited.model_dump(mode="json") if (cited := citations.get(ref)) is not None else None,
        )
        for ref in dict.fromkeys(provenance_refs)
    )


def _source_names(sources: Sequence[SourceProvenance]) -> list[str]:
    return list(dict.fromkeys(source.name for source in sources))


def cited_source_names(
    provenance_refs: Sequence[str], citations: Mapping[str, CitationSource]
) -> list[str]:
    """The distinct cited names of `provenance_refs` in first-seen order. A ref the citation
    lacks stands as its own name. A version's bundle headings and download file names derive
    from these names."""
    return _source_names(_sources(provenance_refs, citations))


def stored_artifact_file_name(location: StoredArtifactLocation, *, manifest: Manifest) -> str:
    """Name a linked artifact from its sources' cited names, `prepared_at` and version, as the
    mint names it.

    A definition's sources are its citation's. A collection artifact's are every member's
    `provenance_refs` in member order. A bare collection member artifact takes its member's
    own refs, content version and `prepared_at`. A ref the citation lacks stands as its name.
    """
    content_version = manifest.content_version
    prepared_at = manifest.prepared_at
    if manifest.collection is None:
        refs = [source.ref for source in manifest.citation.sources]
    else:
        refs = [ref for member in manifest.collection.members for ref in member.provenance_refs]
    member_spec_id = location.rendering.get(MEMBER_RENDERING_KEY)
    if manifest.collection is not None and isinstance(member_spec_id, str):
        member = next(
            (m for m in manifest.collection.members if m.definition.spec_id == member_spec_id),
            None,
        )
        if member is not None:
            refs = member.provenance_refs
            content_version = member.content_version
            prepared_at = member.prepared_at
    return artifact_file_name(
        source_names=cited_source_names(refs, manifest.citation.sources_by_ref()),
        prepared_at=prepared_at,
        content_version=content_version,
        media_type=location.media_type,
        rendering=location.rendering,
        statistics=is_statistics_rendering(location.rendering),
    )


def sign_stored_artifact(
    location: StoredArtifactLocation,
    *,
    manifest: Manifest,
    artifact_store: ArtifactStore,
    read_at: datetime,
) -> ArtifactEntryModel:
    """The manifest entry of one linked artifact location, signed under its
    `stored_artifact_file_name`. Raises `RenderedArtifactSigningFailed` when signing fails."""
    url = _signed_url(
        location.artifact_digest,
        tier=Tier(location.tier),
        media_type=location.media_type,
        file_name=stored_artifact_file_name(location, manifest=manifest),
        artifact_store=artifact_store,
    )
    return ArtifactEntryModel(
        artifact_digest=location.artifact_digest,
        rendering=dict(location.rendering),
        bytes=location.size_bytes,
        media_type=location.media_type,
        # First-sighting time, as a snapshot read lists it.
        created_at=location.created_at.isoformat(),
        tier=location.tier,
        url=url,
        url_expires_at=(read_at + SIGNED_URL_ADVERTISED_TTL).isoformat(),
    )


def metadata_deadline(deadline: float | None, *, floor_seconds: float) -> float | None:
    """The absolute monotonic deadline of one graph metadata read starting now: `deadline`, or
    `floor_seconds` from now when that is later. `None` stays unbounded."""
    if deadline is None:
        return None
    return max(deadline, time.monotonic() + floor_seconds)


def _data_paths(
    member_sources: Sequence[Sequence[SourceProvenance]], *, format_: str
) -> tuple[str, ...]:
    """Each member's data file path, `data/<slug>.<format>`, slugged from its sources with
    `bundle_naming.member_slugs`, which keeps slugs distinct and clear of the codes table."""
    slugs = member_slugs([_source_names(sources) for sources in member_sources])
    return tuple(f"data/{slug}.{format_}" for slug in slugs)


def _sources_heading(names: Sequence[str], window: tuple[int, int] | None) -> str:
    if not names:
        return "Data"
    start, end = window if window is not None else (None, None)
    return heading(source_names=names, start_year=start, end_year=end)


def _member_provenance(
    *,
    spec_id: str,
    written: WrittenArtifact,
    frame: Table,
    rendering: Rendering,
    data_path: str,
    sources: Sequence[SourceProvenance],
    scan: TableScan,
    options: MintOptions,
    code_index: CodeIndex | None,
    row_count: bool,
) -> MemberProvenance:
    """One bundle member, titled by its sources and years, with each column documented from
    `scan`. `row_count` sets the member's row count, which collection bundles print."""
    entry = BundleArtifactEntry(
        path=data_path,
        bytes=written.size_bytes,
        digest=written.digest,
        media_type=written.media_type,
        table_schema=table_schema_for(frame, rendering=rendering),
    )
    code_system_names = (
        {name: declaration.plain_name for name, declaration in code_index.declarations.items()}
        if code_index is not None
        else {}
    )
    return MemberProvenance(
        spec_id=spec_id,
        title=_sources_heading(_source_names(sources), year_window((scan,))),
        artifact=entry,
        sources=tuple(sources),
        columns=column_docs(scan, code_system_names=code_system_names, rendering=rendering),
        # A pivot spreads observations across columns, so only an unpivoted file holds one per row.
        description="One row per observation." if options.pivot is None else None,
        row_count=scan.rows if row_count else None,
    )


def _table_scan(role_columns: Sequence[str], *, options: MintOptions) -> TableScan:
    """A scan of one data file written under `options`, told the pivot that shaped it."""
    pivot = options.pivot
    return TableScan(
        role_columns=role_columns,
        pivot_index=pivot.index if pivot is not None else None,
        pivot_columns=pivot.columns if pivot is not None else None,
    )


def _codes(
    scans: Sequence[TableScan],
    *,
    citations: Mapping[str, CitationSource],
    family: tuple[str, str] | None,
    enrichment: EnrichmentLookup,
    deadline: float | None,
    metadata_read_seconds: float,
) -> BundleCodes:
    """The codes table of the scanned files. Cited sources take their cited names, and a
    collapsed family's `(ref, label)` takes its label. The graph names every other code, within
    `deadline` or `metadata_read_seconds` from now, whichever ends later."""
    known = {ref: cited.label for ref, cited in citations.items()}
    if family is not None:
        known[family[0]] = family[1]
    return bundle_codes(
        scans,
        known_labels=known,
        lookup=functools.partial(
            enrichment.lookup,
            deadline=metadata_deadline(deadline, floor_seconds=metadata_read_seconds),
        ),
    )


def _family(
    collapse: GroupbyCollapseRendering | None, *, subject_id: str, subject_label: str | None
) -> tuple[str, str] | None:
    """The `(ref, label)` of the family ref a summed collapse writes in `stat_var`, `None`
    without one or without a label."""
    if collapse is None or subject_label is None:
        return None
    if collapse.stat_var_ref != f"{_STAT_VAR_REF_PREFIX}{subject_id}":
        return None
    return collapse.stat_var_ref, subject_label


def _rebuild_request(spec: Mapping[str, object], options: MintOptions) -> dict[str, object]:
    """The `POST /api/datasets` body that mints a standalone definition again: its `spec_id`
    with the canonical definition, and these options."""
    return {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": spec.get("spec_id"), "definition": dict(spec)},
        "options": options.model_dump(mode="json", exclude_none=True),
    }


def _site_replayable(spec: Mapping[str, object]) -> bool:
    """Whether the website can open `spec`: its identity is a described arm, the only arm
    Search's `dataset_resubmit` replays."""
    return pinned_member_ref(cast("Mapping[str, object]", spec.get("identity", {}))) is None


def _bundle_context(
    *,
    spec: Mapping[str, object],
    content_version: str,
    members: Sequence[MemberProvenance],
    scans: Sequence[TableScan],
    codes: Sequence[CodeEntry],
    options: MintOptions,
    prepared_at: datetime,
    site: PublicSiteSettings,
) -> BundleContext:
    """The bundle context of one version: headed and named by its distinct sources and years,
    summarised from `scans`, and linked under the public site. Only a standalone definition
    gets a rebuild request. The website link (when `site.site_dataset_links` is on) goes only to
    a standalone definition with a described identity: the website replays the identity through
    Search's `dataset_resubmit`, which refuses the pinned arm (`kind: "stat_var"`) that
    `statvar_set` and `source_codes` mint."""
    is_collection = "collection_id" in spec
    subject_id = str(spec.get("collection_id") if is_collection else spec.get("spec_id"))
    names = _source_names([source for member in members for source in member.sources])
    origin = site.public_site_url
    site_link = (
        f"{origin}/data?{urlencode({'spec': subject_id, 'v': content_version})}"
        if site.site_dataset_links and not is_collection and _site_replayable(spec)
        else None
    )
    return BundleContext(
        heading=_sources_heading(names, year_window(scans)),
        summary=summary_line(scans, codes=codes),
        slug=sources_slug(names),
        version=content_version,
        prepared_at=prepared_at,
        site_url=origin,
        links=BundleLinks(
            version=f"{origin}/api/datasets/{subject_id}/{content_version}",
            latest=f"{origin}/api/datasets/{subject_id}",
            site=site_link,
        ),
        rebuild_request=None if is_collection else _rebuild_request(spec, options),
    )


def _store_artifact(
    *,
    data: bytes,
    media_type: str,
    rendering: Mapping[str, object],
    subject_id: str,
    content_version: str,
    source_names: Sequence[str],
    prepared_at: datetime,
    artifact_store: ArtifactStore,
) -> ArtifactEntryModel:
    with phase("store"):
        try:
            stored = artifact_store.put(data, tier=Tier.working)
        except ArtifactRecordConflictError as exc:
            # `ArtifactRecordConflictError` has typed subject/version fields. This call has both
            # values, which the mint classifier reads for its reconciliation log.
            exc.subject_id = subject_id
            exc.content_version = content_version
            raise
    return _commit_stored(
        stored,
        media_type=media_type,
        rendering=rendering,
        subject_id=subject_id,
        content_version=content_version,
        source_names=source_names,
        prepared_at=prepared_at,
        artifact_store=artifact_store,
    )


def _signed_url(
    digest: str, *, tier: Tier, media_type: str, file_name: str, artifact_store: ArtifactStore
) -> str:
    """A signed download URL for stored bytes, naming the download `file_name`. Raises
    `RenderedArtifactSigningFailed` when signing fails."""
    signing_failed = False
    signing_bucket: str | None = None
    signing_object_name: str | None = None
    signing_digest: str | None = None
    signing_error_class_name: str | None = None
    signing_status_code: int | None = None
    try:
        with phase("sign"):
            url = artifact_store.signed_get_url(
                digest,
                tier=tier,
                ttl=SIGNED_URL_TTL,
                response_type=media_type,
                response_disposition=artifact_content_disposition(file_name),
            )
    except ArtifactSigningError as exc:
        # Read safe fields here. Raise after the handler exits so `__context__` stays unset.
        signing_bucket = exc.bucket
        signing_object_name = exc.object_name
        signing_digest = exc.digest
        signing_error_class_name = exc.error_class_name
        signing_status_code = exc.status_code
        signing_failed = True
    if signing_failed:
        raise RenderedArtifactSigningFailed(
            bucket=cast(str, signing_bucket),
            object_name=cast(str, signing_object_name),
            digest=cast(str, signing_digest),
            error_class_name=signing_error_class_name,
            status_code=signing_status_code,
        )
    return url


def _commit_stored(
    stored: StoredArtifact,
    *,
    media_type: str,
    rendering: Mapping[str, object],
    subject_id: str,
    content_version: str,
    source_names: Sequence[str],
    prepared_at: datetime,
    artifact_store: ArtifactStore,
) -> ArtifactEntryModel:
    """Signs a committed `put` or `open_put` result and builds its `ArtifactEntryModel`. Streaming
    writes pass the committed sink result directly, avoiding a whole-artifact buffer. The signed
    URL names the download from `source_names` and `prepared_at`
    (`artifact_delivery.artifact_file_name`)."""
    file_name = artifact_file_name(
        source_names=source_names,
        prepared_at=prepared_at,
        content_version=content_version,
        media_type=media_type,
        rendering=rendering,
        statistics=is_statistics_rendering(rendering),
    )
    url = _signed_url(
        stored.digest,
        tier=Tier.working,
        media_type=media_type,
        file_name=file_name,
        artifact_store=artifact_store,
    )
    now = datetime.now(UTC)
    return ArtifactEntryModel(
        artifact_digest=stored.digest,
        rendering=dict(rendering),
        bytes=stored.size_bytes,
        media_type=media_type,
        created_at=now.isoformat(),
        tier="working",
        url=url,
        url_expires_at=(now + SIGNED_URL_ADVERTISED_TTL).isoformat(),
    )


def _reshape_needs_whole_table(options: MintOptions) -> bool:
    """Returns true for groupby and pivot, which read across rows. Decode, selection, and
    enrichment are row-wise and can stream."""
    return options.groupby is not None or options.pivot is not None


def _merge_enrichment_warnings(
    warnings: Sequence[MaterialisationWarning],
    *,
    order: Sequence[str],
) -> tuple[MaterialisationWarning, ...]:
    """Combines `enrichment_unavailable` warnings by column, sums their counts, and orders them by
    the decode and enrichment request order. This gives chunked rendering the same counts and order
    as a whole-table reshape. Columns outside `order` follow first-seen order. Other warning
    types pass through after merged entries."""
    merged: dict[str, EnrichmentUnavailableWarning] = {}
    first_seen: list[str] = []
    passthrough: list[MaterialisationWarning] = []
    for warning in warnings:
        if warning["code"] != "enrichment_unavailable":
            passthrough.append(warning)
            continue
        entry = cast(EnrichmentUnavailableWarning, warning)
        column = entry["column"]
        if column not in merged:
            merged[column] = entry
            first_seen.append(column)
        else:
            existing = merged[column]
            merged[column] = EnrichmentUnavailableWarning(
                code="enrichment_unavailable",
                severity=existing["severity"],
                message=existing["message"],
                column=column,
                count=existing["count"] + entry["count"],
            )
    sorted_columns = [column for column in order if column in merged]
    sorted_columns.extend(column for column in first_seen if column not in order)
    return (*(merged[column] for column in sorted_columns), *passthrough)


def _stream_bare_pivot(
    frame_chunks: Iterable[Table],
    *,
    options: MintOptions,
    entity_columns: Sequence[str],
    enrichment: EnrichmentLookup,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None,
    resolved_stat_var_refs: Sequence[str],
    provenance_refs: Sequence[str],
    subject_id: str,
    groupby_preimage: MintPreimage | None,
) -> ReshapeResult:
    """Folds pivot output from chunked rows in one pass. Decode and selection run per chunk, and
    the output stays within `column_cap` and `cell_cap`.

    A groupby that collapses nothing still validates `by`. When `stat_var` is omitted, multiple
    contributors require a decoded summable axis. This refusal runs before pivot, as in whole-table
    processing. `distinct_dcids` gathers the required refs during the fold.

    Enrichment lookups collect per-ref row counts for whole-table missing-value warnings. Pivot
    output carries the index and pivoted values.
    """
    frame_chunks_iter = iter(frame_chunks)
    first_chunk = next(frame_chunks_iter, None)
    assert first_chunk is not None, (
        "_stream_bare_pivot requires frame_chunks to yield at least one chunk"
    )
    rest_of_frame_chunks = itertools.chain((first_chunk,), frame_chunks_iter)

    if options.groupby is not None and code_index is not None:
        refuse_unknown_by_axes(by=options.groupby.by, declarations=code_index.declarations)

    decode_plan = (
        build_decode_plan(first_chunk, index=code_index, extra_refs=resolved_stat_var_refs)
        if code_index is not None
        else None
    )
    decoded_column_names = (
        tuple(
            f"{entity_column}_{code_system}" for entity_column, code_system in decode_plan.requests
        )
        if decode_plan is not None
        else ()
    )
    post_decode_columns = (*first_chunk.columns, *decoded_column_names)

    parsed: ParsedColumns | None = None
    if options.columns is not None:
        try:
            parsed = parse_column_tokens(
                options.columns,
                table_columns=post_decode_columns,
                entity_columns=entity_columns,
                supported_fields=SUPPORTED_ENRICHMENTS,
            )
        except ColumnUnknownError as exc:
            exc.pointer = "/options/columns"
            exc.alternatives = [
                Alternative(ref=column, label=column) for column in post_decode_columns
            ] + list(exc.alternatives or [])
            raise
        disabled = sorted(
            {
                field
                for _entity_column, field in parsed.enrichments
                if field not in ENABLED_ENRICHMENTS
            }
        )
        if disabled:
            raise CapabilityUnavailable(
                detail=f"enrichment field(s) {disabled!r} are recognised but not enabled on "
                "this deployment; GET /api/datasets/capabilities names the enrichment fields "
                "that are live",
                pointer="/options/columns",
            )

    # Classify pre-decode columns, matching the whole-table path.
    classification = classify_columns(first_chunk.columns)
    select = (parsed.select or None) if parsed is not None else None
    assert options.pivot is not None, "_stream_bare_pivot requires options.pivot to be set"
    pivot_plan = PivotPlan(
        index=options.pivot.index,
        columns=options.pivot.columns,
        column_cap=_column_cap(options.format),
        cell_cap=pivot_max_output_cells,
    )

    warnings: list[MaterialisationWarning] = []
    distinct_dcids: set[str] = set()
    stat_var_pos: list[int] = []

    enrichment_entity_columns = (
        sorted({entity_column for entity_column, _field in parsed.enrichments})
        if parsed is not None
        else ()
    )
    # Count rows per distinct ref and requested entity column for missing-value warnings.
    enrichment_ref_counts: dict[str, dict[str, int]] = {ec: {} for ec in enrichment_entity_columns}
    # Non-string entity cells count as lookup misses for every field. Count them once per entity
    # column, separate from the ref counts.
    enrichment_non_string_counts: dict[str, int] = dict.fromkeys(enrichment_entity_columns, 0)

    def _decoded_selected_chunks() -> Iterator[Table]:
        for chunk in rest_of_frame_chunks:
            if decode_plan is not None:
                decoded, decode_warnings = apply_reshapes(
                    chunk, plan=ReshapePlan(decode=decode_plan)
                )
                decoded = cast(Table, decoded)
                warnings.extend(decode_warnings)
            else:
                decoded = chunk
            if options.groupby is not None:
                if not stat_var_pos:
                    stat_var_pos.append(decoded.columns.index(_STAT_VAR_COLUMN))
                pos = stat_var_pos[0]
                for row in decoded.rows:
                    cell = row[pos]
                    if isinstance(cell, str):
                        distinct_dcids.add(cell)
            for entity_column in enrichment_entity_columns:
                entity_pos = decoded.columns.index(entity_column)
                counts = enrichment_ref_counts[entity_column]
                for row in decoded.rows:
                    ref = row[entity_pos]
                    if isinstance(ref, str):
                        counts[ref] = counts.get(ref, 0) + 1
                    else:
                        enrichment_non_string_counts[entity_column] += 1
            if select is not None:
                decoded = select_columns(decoded, keep=select, classification=classification)
            yield decoded

    try:
        outcome = fold_pivot_chunks(
            _decoded_selected_chunks(),
            index=pivot_plan.index,
            columns=pivot_plan.columns,
            column_cap=pivot_plan.column_cap,
            cell_cap=pivot_plan.cell_cap,
            classification=classification,
            value_column=pivot_plan.value_column,
        )
    except ColumnUnknownError as exc:
        exc.pointer = "/options/pivot"
        raise

    if options.groupby is not None and code_index is not None:
        decoded_code_systems = (
            tuple(code_system for _entity_column, code_system in decode_plan.requests)
            if decode_plan is not None
            else ()
        )
        if (
            _STAT_VAR_COLUMN not in options.groupby.by
            and not declared_axes_present(
                decoded_code_systems=decoded_code_systems,
                declarations=code_index.declarations,
            )
            and len(distinct_dcids) > 1
        ):
            raise RollupNotSummableError(
                f"groupby.by omits {_STAT_VAR_COLUMN!r} but this frame decodes no "
                "declared-summable axis to collapse its contributing StatVars over; "
                "this family's members carry no summable axis, so they cannot be summed",
                axis=_STAT_VAR_COLUMN,
                precondition="no_summable_axis",
            )

    if outcome.violation is not None:
        raise outcome.violation
    assert outcome.table is not None
    pivoted = outcome.table

    enrichment_warnings: tuple[MaterialisationWarning, ...] = ()
    if enrichment_entity_columns:
        assert parsed is not None
        all_refs = sorted({ref for counts in enrichment_ref_counts.values() for ref in counts})
        lookup = enrichment.lookup(tuple(all_refs))
        missing_counts: dict[tuple[str, str], int] = {}
        for entity_column, field_name in parsed.enrichments:
            total_missing = enrichment_non_string_counts.get(entity_column, 0)
            for ref, row_count in enrichment_ref_counts.get(entity_column, {}).items():
                if lookup.get(ref, {}).get(field_name, "") == "":
                    total_missing += row_count
            if total_missing:
                missing_counts[(entity_column, field_name)] = total_missing
        enrichment_warnings = tuple(
            enrichment_unavailable_warning(
                column=f"{entity_column}_{field_name}",
                count=missing_counts[(entity_column, field_name)],
            )
            for entity_column, field_name in parsed.enrichments
            if (entity_column, field_name) in missing_counts
        )

    merged_decode_warnings = _merge_enrichment_warnings(warnings, order=decoded_column_names)
    return ReshapeResult(
        table=cast(Table, pivoted),
        warnings=(*merged_decode_warnings, *enrichment_warnings),
        collapse=None,
        warning_order=(),
    )


@dataclass(frozen=True)
class _StreamedRender:
    """Carries writer chunks and their result callback. Call `finish` after the writer drains. It
    reads accumulated table and warnings and may raise a deferred `no_summable_axis` refusal.
    Callers run it inside `open_put` so that refusal aborts the artifact."""

    chunks_for_writer: Iterable[Table]
    collapse: GroupbyCollapseRendering | None
    finish: Callable[[], tuple[Table, tuple[MaterialisationWarning, ...]]]


def _reshape_stream_plain(
    frame_chunks: Iterable[Table],
    *,
    options: MintOptions,
    entity_columns: Sequence[str],
    enrichment: EnrichmentLookup,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None,
    resolved_stat_var_refs: Sequence[str],
    provenance_refs: Sequence[str],
) -> _StreamedRender:
    """Streams row-wise decode, selection, and enrichment by reshaping each chunk before passing it
    to the writer. `_merge_enrichment_warnings` combines per-chunk counts."""
    collected_warnings: list[MaterialisationWarning] = []
    first_reshaped_chunk: list[Table] = []
    first_warning_order: list[tuple[str, ...]] = []

    def _reshaped_chunks() -> Iterator[Table]:
        for raw_chunk in frame_chunks:
            chunk_result = _reshape(
                raw_chunk,
                options=options,
                entity_columns=entity_columns,
                enrichment=enrichment,
                pivot_max_output_cells=pivot_max_output_cells,
                code_index=code_index,
                resolved_stat_var_refs=resolved_stat_var_refs,
                provenance_refs=provenance_refs,
            )
            collected_warnings.extend(chunk_result.warnings)
            if not first_reshaped_chunk:
                first_reshaped_chunk.append(chunk_result.table)
                first_warning_order.append(chunk_result.warning_order)
            yield chunk_result.table

    def _finish() -> tuple[Table, tuple[MaterialisationWarning, ...]]:
        assert first_reshaped_chunk, (
            "render_and_store requires frame_chunks to yield at least one chunk"
        )
        return (
            first_reshaped_chunk[0],
            _merge_enrichment_warnings(collected_warnings, order=first_warning_order[0]),
        )

    return _StreamedRender(chunks_for_writer=_reshaped_chunks(), collapse=None, finish=_finish)


def _reshape_stream(
    frame_chunks: Iterable[Table],
    *,
    options: MintOptions,
    entity_columns: Sequence[str],
    enrichment: EnrichmentLookup,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None,
    resolved_stat_var_refs: Sequence[str],
    provenance_refs: Sequence[str],
    subject_id: str,
    subject_label: str | None,
    groupby_preimage: MintPreimage | None,
) -> _StreamedRender:
    """Reslices chunks, then dispatches every member through one of four paths:

    - Plain decode, selection, and enrichment stream each reshaped chunk to the writer.
    - A collapsing groupby folds `stat_var` across chunks, then writes the reduced table.
    - A bare pivot folds output within its width and cell caps.
    - A no-op groupby streams per chunk and checks its refusal after the writer drains.

    The same dispatch serves definitions and collection members. Groupby and pivot folds bound
    output size. Other paths retain chunk-sized row data.
    """
    frame_chunks_iter = iter(_reslice_chunks(frame_chunks))
    first_chunk = next(frame_chunks_iter, None)
    rest_of_frame_chunks = (
        itertools.chain((first_chunk,), frame_chunks_iter) if first_chunk is not None else iter(())
    )

    if not _reshape_needs_whole_table(options):
        return _reshape_stream_plain(
            rest_of_frame_chunks,
            options=options,
            entity_columns=entity_columns,
            enrichment=enrichment,
            pivot_max_output_cells=pivot_max_output_cells,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            provenance_refs=provenance_refs,
        )

    will_stream_groupby_collapse = False
    probe_decoded_code_systems: tuple[str, ...] = ()
    if first_chunk is not None and options.groupby is not None and code_index is not None:
        probe_decode_plan = build_decode_plan(
            first_chunk, index=code_index, extra_refs=resolved_stat_var_refs
        )
        probe_decoded_code_systems = (
            tuple(code_system for _entity_column, code_system in probe_decode_plan.requests)
            if probe_decode_plan is not None
            else ()
        )
        will_stream_groupby_collapse = bool(
            collapse_axes(
                by=options.groupby.by,
                decoded_code_systems=probe_decoded_code_systems,
                declarations=code_index.declarations,
            )
        )

    if will_stream_groupby_collapse and first_chunk is not None:
        # `_reshape` recomputes this decision from the same inputs. A chained pivot runs on the
        # reduced output groups.
        result = _reshape(
            first_chunk,
            options=options,
            entity_columns=entity_columns,
            enrichment=enrichment,
            pivot_max_output_cells=pivot_max_output_cells,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            provenance_refs=provenance_refs,
            subject_id=subject_id,
            subject_label=subject_label,
            groupby_preimage=groupby_preimage,
            frame_chunks=rest_of_frame_chunks,
        )
        return _StreamedRender(
            chunks_for_writer=(result.table,),
            collapse=result.collapse,
            finish=lambda: (result.table, result.warnings),
        )

    if first_chunk is not None and options.pivot is not None and not will_stream_groupby_collapse:
        # Stream bare pivots and pivots after a no-op groupby. The fold checks that groupby families
        # with multiple contributors include a summable axis.
        result = _stream_bare_pivot(
            rest_of_frame_chunks,
            options=options,
            entity_columns=entity_columns,
            enrichment=enrichment,
            pivot_max_output_cells=pivot_max_output_cells,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            provenance_refs=provenance_refs,
            subject_id=subject_id,
            groupby_preimage=groupby_preimage,
        )
        return _StreamedRender(
            chunks_for_writer=(result.table,),
            collapse=result.collapse,
            finish=lambda: (result.table, result.warnings),
        )

    if options.groupby is not None and code_index is not None:
        refuse_unknown_by_axes(by=options.groupby.by, declarations=code_index.declarations)
    passthrough_options = (
        options.model_copy(update={"groupby": None}) if options.groupby is not None else options
    )
    collected_warnings: list[MaterialisationWarning] = []
    first_reshaped_chunk: list[Table] = []
    first_warning_order: list[tuple[str, ...]] = []
    distinct_dcids: set[str] = set()
    stat_var_pos: list[int] = []

    def _groupby_passthrough_chunks() -> Iterator[Table]:
        for raw_chunk in rest_of_frame_chunks:
            if options.groupby is not None and code_index is not None:
                if not stat_var_pos and _STAT_VAR_COLUMN in raw_chunk.columns:
                    stat_var_pos.append(raw_chunk.columns.index(_STAT_VAR_COLUMN))
                if stat_var_pos:
                    pos = stat_var_pos[0]
                    for row in raw_chunk.rows:
                        cell = row[pos]
                        if isinstance(cell, str):
                            distinct_dcids.add(cell)
            chunk_result = _reshape(
                raw_chunk,
                options=passthrough_options,
                entity_columns=entity_columns,
                enrichment=enrichment,
                pivot_max_output_cells=pivot_max_output_cells,
                code_index=code_index,
                resolved_stat_var_refs=resolved_stat_var_refs,
                provenance_refs=provenance_refs,
            )
            collected_warnings.extend(chunk_result.warnings)
            if not first_reshaped_chunk:
                first_reshaped_chunk.append(chunk_result.table)
                first_warning_order.append(chunk_result.warning_order)
            yield chunk_result.table

    def _finish() -> tuple[Table, tuple[MaterialisationWarning, ...]]:
        if (
            options.groupby is not None
            and code_index is not None
            and _STAT_VAR_COLUMN not in options.groupby.by
            and not declared_axes_present(
                decoded_code_systems=probe_decoded_code_systems,
                declarations=code_index.declarations,
            )
            and len(distinct_dcids) > 1
        ):
            raise RollupNotSummableError(
                f"groupby.by omits {_STAT_VAR_COLUMN!r} but this frame decodes no "
                "declared-summable axis to collapse its contributing StatVars over; "
                "this family's members carry no summable axis, so they cannot be summed",
                axis=_STAT_VAR_COLUMN,
                precondition="no_summable_axis",
            )
        assert first_reshaped_chunk, (
            "render_and_store requires frame_chunks to yield at least one chunk"
        )
        merged = _merge_enrichment_warnings(collected_warnings, order=first_warning_order[0])
        return first_reshaped_chunk[0], merged

    return _StreamedRender(
        chunks_for_writer=_groupby_passthrough_chunks(), collapse=None, finish=_finish
    )


@dataclass(frozen=True)
class RenderedDefinition:
    """The `render_and_store` result. `collapse` is set for a full groupby collapse.
    `datasets_api.serve` uses it to write the collapsed version's frame1 and version record.
    `labels_complete` is `False` when a bundle's code label read failed
    (`BundleCodes.label_read_complete`), and the caller then leaves the artifact unlinked."""

    artifact: ArtifactEntryModel
    warnings: tuple[MaterialisationWarning, ...]
    collapse: GroupbyCollapseRendering | None
    labels_complete: bool = True


def render_and_store(
    *,
    frame_chunks: Iterable[Table],
    options: MintOptions,
    spec: dict[str, object],
    provenance_refs: list[str],
    citations: Mapping[str, CitationSource],
    reproducing_call: ReproducingCall | None,
    content_version: str,
    prepared_at: datetime,
    site: PublicSiteSettings,
    artifact_store: ArtifactStore,
    entity_columns: Sequence[str],
    enrichment: EnrichmentLookup,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None = None,
    resolved_stat_var_refs: Sequence[str],
    groupby_preimage: MintPreimage | None = None,
    deadline: float | None = None,
    metadata_read_seconds: float = 0.0,
) -> RenderedDefinition:
    """Renders one definition to a stored artifact and returns its reshape warnings. It consumes
    `frame_chunks` once in order. `_reshape_stream` shares dispatch with collection members.

    A bundle is built from the version's facts: `citations` (the version's
    `CitationModel.sources_by_ref()`, with empty documentation and licence fields filled by
    `datasets_api.manifest.healed_citation`) names and documents each source, and `prepared_at`
    is the version's first-mint instant. Its data file is `data/<sources slug>.<format>`, its
    columns are documented and its codes listed from the rows the writer streams, and the codes
    are named by one batched graph read through `enrichment`, bounded by `deadline` or
    `metadata_read_seconds` from the read's start, whichever ends later. A fresh render of one
    version therefore matches the artifact linked for it while the graph's documentation and
    code names are unchanged. `site` supplies the origin of every link.

    Decode, selection, and enrichment stream one chunk at a time. Enrichment warnings merge into
    one count per column. Groupby folds across chunks. Pivot folds to bounded output. No-op groupby
    also streams, so wide results can exceed 1,000,000 rows. The raw-fetch guard allows 1.5 times
    the coverage estimate.

    `code_index` controls decoding. `resolved_stat_var_refs` and `provenance_refs` support groupby
    population and source checks. `groupby_preimage` can mint a full-collapse version. Its artifact
    uses that version when present and `content_version` otherwise. `serve` writes version records.
    A full collapse's `stat_vars_summed` warning labels the family with
    `code_index.labels.family_label` over `spec`'s identity.
    """
    check_options_supported(options)
    subject_id = str(spec.get("spec_id") or spec.get("collection_id"))
    identity = spec.get("identity")
    subject_label = (
        family_label(
            cast("Mapping[str, object]", identity),
            index=code_index,
            measure_index=get_measure_index(),
        )
        if code_index is not None and isinstance(identity, Mapping)
        else None
    )
    with phase("render"):
        rendering = Rendering()
        streamed = _reshape_stream(
            frame_chunks,
            options=options,
            entity_columns=entity_columns,
            enrichment=enrichment,
            pivot_max_output_cells=pivot_max_output_cells,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            provenance_refs=provenance_refs,
            subject_id=subject_id,
            subject_label=subject_label,
            groupby_preimage=groupby_preimage,
        )
        chunks_for_writer = streamed.chunks_for_writer
        collapse = streamed.collapse
        effective_content_version = (
            collapse.content_version
            if collapse is not None and collapse.content_version is not None
            else content_version
        )

    sources = _sources(provenance_refs, citations)
    with phase("store"):
        try:
            with artifact_store.open_put(tier=Tier.working) as sink:
                if options.bundle:
                    # Streamed: the member's own data entry is written
                    # first (`force_zip64=True` since its size is unknown
                    # until the writer finishes -- see `artifact_store
                    # .store.ArtifactSink`'s own never-seekable shape, which
                    # is what makes `zf`'s own `_seekable` probe land False
                    # and every entry use a data descriptor regardless), so
                    # `datapackage.json` (built from the member's own
                    # digest/size, only known afterward) can be written
                    # last, into the same still-open archive.
                    # `ArtifactSink` provides `write`/`tell` and checks seekability at construction.
                    # Cast it to `IO[bytes]` for `zipfile`'s nominal parameter type.
                    with zipfile.ZipFile(cast(IO[bytes], sink), "w") as zf:
                        (data_path,) = _data_paths((sources,), format_=options.format)
                        scan = _table_scan(entity_columns, options=options)
                        with zf.open(
                            zipfile.ZipInfo(data_path, date_time=_ZIP_FIXED_DATE_TIME),
                            mode="w",
                            force_zip64=True,
                        ) as entry:
                            written = _FORMAT_WRITERS[options.format](
                                scan.observe(chunks_for_writer), rendering=rendering, sink=entry
                            )
                        # `streamed.finish()` runs while `sink`/`zf` are
                        # still open, so a refusal it raises (the deferred
                        # `no_summable_axis` check) still aborts this write
                        # -- see `_StreamedRender`'s own docstring.
                        table, warnings = streamed.finish()
                        bundle_codes_read = _codes(
                            (scan,),
                            citations=citations,
                            family=_family(
                                collapse, subject_id=subject_id, subject_label=subject_label
                            ),
                            enrichment=enrichment,
                            deadline=deadline,
                            metadata_read_seconds=metadata_read_seconds,
                        )
                        labels_complete = bundle_codes_read.label_read_complete
                        codes = bundle_codes_read.entries
                        member = _member_provenance(
                            spec_id=subject_id,
                            written=written,
                            frame=table,
                            rendering=rendering,
                            data_path=data_path,
                            sources=sources,
                            scan=scan,
                            options=options,
                            code_index=code_index,
                            row_count=False,
                        )
                        files = assemble_bundle(
                            spec=spec,
                            members=(member,),
                            reproducing_call=reproducing_call,
                            context=_bundle_context(
                                spec=spec,
                                content_version=effective_content_version,
                                members=(member,),
                                scans=(scan,),
                                codes=codes,
                                options=options,
                                prepared_at=prepared_at,
                                site=site,
                            ),
                            codes=codes,
                        )
                        for name in sorted(files):
                            zf.writestr(
                                zipfile.ZipInfo(name, date_time=_ZIP_FIXED_DATE_TIME),
                                files[name],
                            )
                    media_type = "application/zip"
                else:
                    written = _FORMAT_WRITERS[options.format](
                        chunks_for_writer, rendering=rendering, sink=sink
                    )
                    table, warnings = streamed.finish()
                    media_type = written.media_type
                    labels_complete = True
            stored = sink.result()
        except ArtifactRecordConflictError as exc:
            # See `_store_artifact`'s own identical clause: this call sites
            # the same conflict context (`subject_id`/`content_version`)
            # onto the exception before it propagates.
            exc.subject_id = subject_id
            exc.content_version = effective_content_version
            raise

    artifact = _commit_stored(
        stored,
        media_type=media_type,
        rendering=artifact_rendering(options, site=site),
        subject_id=subject_id,
        content_version=effective_content_version,
        source_names=_source_names(sources),
        prepared_at=prepared_at,
        artifact_store=artifact_store,
    )
    return RenderedDefinition(
        artifact=artifact, warnings=warnings, collapse=collapse, labels_complete=labels_complete
    )


MEMBER_RENDERING_KEY = "member"
"""Tags each bare collection artifact with its member `spec_id`, keeping rendering keys distinct
under the shared subject and version. Snapshot retrieval removes this key before applying request
rendering filters."""

BUNDLE_LAYOUT_VERSION = 1
"""The bundle layout `rendition_token` hashes. Bump it with any change to a bundle's files or
their contents for unchanged data and settings (a file added, renamed or reworded, a column or
code described differently), so every version renders and links a new bundle on its next
request while each old bundle stays listed under its own rendering key."""

RENDITION_KEY = "rendition"
"""The bundle rendering's rendition token (`rendition_token`). A bare rendering has none."""


def rendition_token(site: PublicSiteSettings) -> str:
    """`v<BUNDLE_LAYOUT_VERSION>-` and 12 hex characters of a SHA-256 over the bundle layout
    version, `site.public_site_url` and `site.site_dataset_links`, the settings a bundle's
    bytes depend on besides its version and options."""
    canonical = json.dumps(
        [BUNDLE_LAYOUT_VERSION, site.public_site_url, site.site_dataset_links],
        separators=(",", ":"),
    )
    return f"v{BUNDLE_LAYOUT_VERSION}-{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"


def artifact_rendering(options: MintOptions, *, site: PublicSiteSettings) -> dict[str, object]:
    """The `rendering` an artifact rendered under `options` is stored, linked and matched under.

    It is `options` without unset fields (`exclude_none=True`, so a defaulted `format` or
    `bundle` always appears and an unset reshape never does), plus `RENDITION_KEY` for a bundle.
    The rendering key hashes this dict (`version_store.rendering_key`), so a layout or settings
    change gives new bundles a new key and leaves old links matching only themselves.
    """
    rendering = options.model_dump(mode="json", exclude_none=True)
    if options.bundle:
        rendering[RENDITION_KEY] = rendition_token(site)
    return rendering


@dataclass(frozen=True)
class CollectionMemberRendering:
    """Input for one collection member. `frame_chunks` is called once when that member renders.
    Members render in order, keeping only one member's rows in memory.

    `content_version` is the member's token, separate from the collection version, and
    `prepared_at` its first-mint instant, which names its bare file. `entity_columns`
    are its declared roles, which may differ across members. `provenance_refs` are the member's
    live refs, which govern the member-specific absent-observation rule for groupby.
    `cited_refs` name and cite its files: the bundle's name, the member's `data/` path, its
    README sources and its bare file's name. They are the stored member entry's refs for a
    collection already on record, and equal `provenance_refs` otherwise.
    `resolved_stat_var_refs` includes zero-row members in groupby population checks.
    """

    spec_id: str
    frame_chunks: Callable[[], Iterator[Table]]
    provenance_refs: Sequence[str]
    cited_refs: Sequence[str]
    content_version: str
    prepared_at: datetime
    entity_columns: Sequence[str]
    resolved_stat_var_refs: Sequence[str]
    """This member's own resolved StatVar population, zero-row members
    included -- threaded straight through to `_reshape` (see that
    function's own docstring). A collection's members each materialise
    their own frame independently, so each carries its own population
    rather than sharing one across the collection.
    """


def _reshape_collection_member(
    member: CollectionMemberRendering,
    *,
    options: MintOptions,
    enrichment: EnrichmentLookup,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None,
) -> _StreamedRender:
    """Streams one member's chunks through shared reshape dispatch. A full collapse with multiple
    contributors raises `RollupNotSummableError` because each member uses its base content version.
    The check runs before member bytes are written. Sole-contributor collapses succeed.
    The member id supplies the family ref. The version preimage is absent."""
    streamed = _reshape_stream(
        member.frame_chunks(),
        options=options,
        entity_columns=member.entity_columns,
        enrichment=enrichment,
        pivot_max_output_cells=pivot_max_output_cells,
        code_index=code_index,
        resolved_stat_var_refs=member.resolved_stat_var_refs,
        provenance_refs=member.provenance_refs,
        subject_id=member.spec_id,
        # A collection member refuses a summed collapse below, so no warning needs the label.
        subject_label=None,
        groupby_preimage=None,
    )
    if streamed.collapse is not None and streamed.collapse.contributing_member_count > 1:
        raise RollupNotSummableError(
            f"groupby fully collapses collection member {member.spec_id!r} across "
            f"{streamed.collapse.contributing_member_count} contributing "
            "StatVars; a collection has no per-member content_version to mint the "
            "collapsed identity under -- request this member individually through "
            f"GET /api/datasets/{member.spec_id} instead",
            axis="stat_var",
            precondition="collection_member_family_collapse",
        )
    return streamed


def _stream_member_artifact(
    streamed: _StreamedRender,
    *,
    format_: str,
    rendering: Rendering,
    subject_id: str,
    content_version: str,
    artifact_store: ArtifactStore,
) -> tuple[WrittenArtifact, StoredArtifact, Table, tuple[MaterialisationWarning, ...]]:
    """Streams writer output into `open_put` and calls `finish` before closing the sink, so deferred
    refusals abort the write. Adds `subject_id` and `content_version` to artifact conflicts before
    re-raising them."""
    with phase("store"):
        try:
            with artifact_store.open_put(tier=Tier.working) as sink:
                written = _FORMAT_WRITERS[format_](
                    streamed.chunks_for_writer, rendering=rendering, sink=sink
                )
                table, warnings = streamed.finish()
            stored = sink.result()
        except ArtifactRecordConflictError as exc:
            exc.subject_id = subject_id
            exc.content_version = content_version
            raise
    return written, stored, table, warnings


@dataclass(frozen=True)
class RenderedCollection:
    """The `render_and_store_collection` result: the stored artifacts, the reshape warnings in
    member order, and whether a bundle's code label read answered, as in
    `RenderedDefinition`."""

    artifacts: list[ArtifactEntryModel]
    warnings: tuple[MaterialisationWarning, ...]
    labels_complete: bool = True


def render_and_store_collection(
    *,
    members: Iterable[CollectionMemberRendering],
    options: MintOptions,
    spec: dict[str, object],
    citations: Mapping[str, CitationSource],
    reproducing_call: ReproducingCall | None,
    content_version: str,
    prepared_at: datetime,
    site: PublicSiteSettings,
    artifact_store: ArtifactStore,
    enrichment: EnrichmentLookup,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None = None,
    deadline: float | None = None,
    metadata_read_seconds: float = 0.0,
) -> RenderedCollection:
    """Consumes members once, in order, and renders one at a time. It holds at most one member's
    table and output bytes. The next member starts after the current write commits.

    Bare mode stores one artifact per member under that member's version. `MEMBER_RENDERING_KEY`
    keeps artifacts distinct under the shared collection subject and version. Bundle mode stores one
    collection-version artifact with a resource per member. Its rendering key covers the package.
    Member files stream into the ZIP before `datapackage.json`, whose sizes and digests are known
    after every writer finishes.

    Returns reshape warnings in member order. A supplied `code_index` passes to each member's
    reshape. `None` skips decoding. `citations` is the manifest's `sources_by_ref()`, covering
    every member's refs.

    A bundle writes each member at `data/<member slug>.<format>`
    (`bundle_naming.member_slugs`) beside one codes table for every member, prints each
    member's row count, and takes `prepared_at`, `site`, `deadline` and `metadata_read_seconds` as
    `render_and_store` does. A collection bundle gets no website link and no rebuild request.
    """
    check_options_supported(options)
    rendering = Rendering()
    reshape_warnings: list[MaterialisationWarning] = []
    rendering_options = artifact_rendering(options, site=site)

    if not options.bundle:
        artifacts: list[ArtifactEntryModel] = []
        for member in members:
            with phase("render"):
                streamed = _reshape_collection_member(
                    member,
                    options=options,
                    enrichment=enrichment,
                    pivot_max_output_cells=pivot_max_output_cells,
                    code_index=code_index,
                )
            written, stored, table, member_warnings = _stream_member_artifact(
                streamed,
                format_=options.format,
                rendering=rendering,
                subject_id=member.spec_id,
                content_version=member.content_version,
                artifact_store=artifact_store,
            )
            del streamed
            reshape_warnings.extend(member_warnings)
            artifacts.append(
                _commit_stored(
                    stored,
                    media_type=written.media_type,
                    rendering={**rendering_options, MEMBER_RENDERING_KEY: member.spec_id},
                    subject_id=member.spec_id,
                    content_version=member.content_version,
                    source_names=cited_source_names(member.cited_refs, citations),
                    prepared_at=member.prepared_at,
                    artifact_store=artifact_store,
                )
            )
            del table
        return RenderedCollection(artifacts=artifacts, warnings=tuple(reshape_warnings))

    members = tuple(members)
    member_sources = [_sources(member.cited_refs, citations) for member in members]
    data_paths = _data_paths(member_sources, format_=options.format)
    bundle_members: list[MemberProvenance] = []
    scans: list[TableScan] = []
    try:
        with (
            artifact_store.open_put(tier=Tier.working) as sink,
            zipfile.ZipFile(cast(IO[bytes], sink), "w") as zf,
        ):
            for member, sources, data_path in zip(members, member_sources, data_paths, strict=True):
                with phase("render"):
                    streamed = _reshape_collection_member(
                        member,
                        options=options,
                        enrichment=enrichment,
                        pivot_max_output_cells=pivot_max_output_cells,
                        code_index=code_index,
                    )
                with phase("store"):
                    scan = _table_scan(member.entity_columns, options=options)
                    with zf.open(
                        zipfile.ZipInfo(data_path, date_time=_ZIP_FIXED_DATE_TIME),
                        mode="w",
                        force_zip64=True,
                    ) as entry:
                        written = _FORMAT_WRITERS[options.format](
                            scan.observe(streamed.chunks_for_writer),
                            rendering=rendering,
                            sink=entry,
                        )
                    # Finish before the sink closes so a deferred refusal aborts the bundle.
                    table, member_warnings = streamed.finish()
                    del streamed
                    reshape_warnings.extend(member_warnings)
                    scans.append(scan)
                    bundle_members.append(
                        _member_provenance(
                            spec_id=member.spec_id,
                            written=written,
                            frame=table,
                            rendering=rendering,
                            data_path=data_path,
                            sources=sources,
                            scan=scan,
                            options=options,
                            code_index=code_index,
                            row_count=True,
                        )
                    )
                # See the bare path's identical `del table` above: drops
                # this member's own table before the next member's own
                # `frame_chunks()` is ever called.
                del table
            with phase("store"):
                # A collection refuses a summed collapse, so no member writes a family ref.
                bundle_codes_read = _codes(
                    scans,
                    citations=citations,
                    family=None,
                    enrichment=enrichment,
                    deadline=deadline,
                    metadata_read_seconds=metadata_read_seconds,
                )
                codes = bundle_codes_read.entries
                files = assemble_bundle(
                    spec=spec,
                    members=tuple(bundle_members),
                    reproducing_call=reproducing_call,
                    context=_bundle_context(
                        spec=spec,
                        content_version=content_version,
                        members=bundle_members,
                        scans=scans,
                        codes=codes,
                        options=options,
                        prepared_at=prepared_at,
                        site=site,
                    ),
                    codes=codes,
                )
                for name in sorted(files):
                    zf.writestr(zipfile.ZipInfo(name, date_time=_ZIP_FIXED_DATE_TIME), files[name])
        stored = sink.result()
    except ArtifactRecordConflictError as exc:
        exc.subject_id = str(spec.get("collection_id"))
        exc.content_version = content_version
        raise

    artifact = _commit_stored(
        stored,
        media_type="application/zip",
        rendering=rendering_options,
        subject_id=str(spec.get("collection_id")),
        content_version=content_version,
        source_names=_source_names([source for sources in member_sources for source in sources]),
        prepared_at=prepared_at,
        artifact_store=artifact_store,
    )
    return RenderedCollection(
        artifacts=[artifact],
        warnings=tuple(reshape_warnings),
        labels_complete=bundle_codes_read.label_read_complete,
    )


_FRAME1_RENDERING: Mapping[str, object] = {"format": "frame1"}
"""Fixed rendering key for frame1, independent of request options. Mints with different options
therefore reuse the same `version_artifact` row for byte-identical frame1 data."""


STATISTICS_MEDIA_TYPE = "application/vnd.apache.parquet"
"""Both `series-statistics/1` and requested Parquet use `application/vnd.apache.parquet`. Rendering
metadata identifies statistics with `{"statistics": "series"}`."""

_STATISTICS_RENDERING_KEY = "statistics"
_STATISTICS_RENDERING_VALUE = "series"


def _statistics_rendering(*, member: str | None) -> Mapping[str, object]:
    """Returns the fixed statistics rendering key, adding `MEMBER_RENDERING_KEY` for a collection
    member so its artifact stays distinct under the shared version."""
    rendering: dict[str, object] = {_STATISTICS_RENDERING_KEY: _STATISTICS_RENDERING_VALUE}
    if member is not None:
        rendering[MEMBER_RENDERING_KEY] = member
    return rendering


def is_statistics_rendering(rendering: Mapping[str, object]) -> bool:
    """Matches `series-statistics/1` rendering keys, including collection-member keys. Snapshot
    retrieval uses this to include statistics independently of the requested format."""
    return rendering.get(_STATISTICS_RENDERING_KEY) == _STATISTICS_RENDERING_VALUE


def render_and_store_series_statistics(
    *,
    data: bytes,
    subject_id: str,
    content_version: str,
    source_names: Sequence[str],
    prepared_at: datetime,
    artifact_store: ArtifactStore,
    member: str | None = None,
) -> ArtifactEntryModel:
    """Stores each materialisation's `series-statistics/1` Parquet as a listable, signed artifact,
    independent of the requested format. Manifests list it when `series_count` is set. Collection
    members include their id in the rendering key to stay distinct under the shared version."""
    return _store_artifact(
        data=data,
        media_type=STATISTICS_MEDIA_TYPE,
        rendering=_statistics_rendering(member=member),
        subject_id=subject_id,
        content_version=content_version,
        source_names=source_names,
        prepared_at=prepared_at,
        artifact_store=artifact_store,
    )


FRAME1_MAX_ROWS = 250_000
"""Most rows a frame1 rendering holds. A mint writes frame1 only at or under it, and the publish
door admits no frame above it (`PUBLISH_MAX_ROWS`). A larger version still mints, and
presentation answers `missing_rendering` for it.

The mint streams frame1 into the store: its Python heap peak is about 32 MiB at both 130,000 and
250,000 rows. The bound's cost falls on presentation, where `LiveFrameStore.load` fetches the
whole document and JSON-decodes every row. Measured on CRS-shaped frames, a root mint's frame1
(two declared roles) is 240 bytes a row, and a published frame with two declared roles and four
`stat_var_*` columns is 363 bytes a row. One load of the wider frame adds 178 MiB of peak RSS at
130,000 rows and 420 MiB at 250,000 rows (120 and 298 MiB for the root shape), and the frame
cache then keeps about 233 MiB of the 250,000-row frame resident."""


def render_and_store_frame1(
    *,
    frame_chunks: Iterable[Frame],
    subject_id: str,
    content_version: str,
    artifact_store: ArtifactStore,
) -> ArtifactRecord:
    """Streams the canonical frame1 bytes of `frame_chunks` into the store as an unsigned
    artifact, without holding the whole document. Presentation reads it from the version record
    by media type. Encoding runs inside the shared `store` phase, so when it runs beside the
    requested write that phase sums both threads' time."""
    with phase("store"):
        try:
            with artifact_store.open_put(tier=Tier.working) as sink:
                written = write_frame1(frame_chunks, sink=sink)
            stored = sink.result()
        except ArtifactRecordConflictError as exc:
            exc.subject_id = subject_id
            exc.content_version = content_version
            raise
    return ArtifactRecord(
        artifact_digest=stored.digest,
        rendering=dict(_FRAME1_RENDERING),
        media_type=written.media_type,
    )
