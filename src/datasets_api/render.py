"""Renders materialised frames as CSV, XLSX, or Parquet, optionally packaged as a Frictionless
Data Package. The party-totals cut, decode, groupby, selection, enrichment, and pivot run before
the writer, in that order. Selection, enrichment and pivot preserve the version's
`content_version`. A cut that leaves rows out (`options.without_party_totals`,
`datasets_api.party_totals`) and a groupby collapse to a frame of its own each mint a version of
their own (`DerivedVersion`); a groupby over a cut frame sums the members the cut kept.

Decode runs first and adds `stat_var_{code_system}` columns, so selection and pivot can name them.
`options.columns` resolves a token to a real or decoded column first, then to an enabled
`{entity_column}_{field}` enrichment (for example `recipient_iso3`). Table and decoded columns
select; a list that names none keeps every default column and only adds the enrichments. Unknown
tokens list the parseable qualified tokens as `alternatives`.

Long (unpivoted) output is written in the layout `materialise.projection` defines: each decoded
`stat_var_<code_system>` column becomes a `<code_system>` code column and a `<code_system>_label`
column, each entity, `measurementMethod` and `provenance` gets a `_label` name column (and
`stat_var` one when no axis decodes), `unit` splits into `unit`, `unit_label`, `currency`,
`price_basis` and `base_year`, enrichments sit beside their entity, the order is fixed whatever
order `columns` names, and `status` is left out. Names are cited source labels, else graph
names read per chunk within the render's deadline (`_LongLayout`). A missing name, a failed or
late read, or an axis label two codes share leaves the cell empty and adds
`enrichment_unavailable`. A failed read also links the artifact as incomplete
(`incomplete_rendering`). `measurementMethod_label`
holds only a name that differs from the method's dcid and raises no warning. A selection keeps every
decoded column unless it names some, and `status` and `<role>_label` tokens are accepted and
ignored. Requests, pivot output, frames and collapse frames keep the grammar names, and so
does a bare parquet file requested with `options.layout = "frame"` (`FRAME_LAYOUT`), which is
written as reshaped, with no projection and no name read.

A long CSV or XLSX keeps total rows out of its main table (`_TotalRows`). A row is a total when a
role that can hold a total (`datasets_api.output_options.movable_totals_roles`, the rule the
options document's `long_layout_totals` reads) holds a party the code index declares a `total`.
A role bound to one entity makes no total row, and residuals and `private` parties (a private donor
reporting beside the official ones) stay with the parties. A published whole-population total is a
`total` with a marker for the row's dataset (`PartyRow.whole`); its frame cell is `whole_total` in the years of the marker's window and `total` outside them or where the row has no readable dataset or year. A bundle's CSV writes the totals as a second data file
(`data/<slug>-totals.csv`), an XLSX on a second sheet (`totals`), and a bare CSV leaves them
out. Parquet, pivot output and the frame layout keep every row.

`groupby` and `pivot` read across chunks. Other reshapes stream rows to the writer. A recognised but
disabled enrichment token raises `capability_unavailable`. Unknown tokens raise `column_unknown`.

Every mint also streams the version's canonical frame to the store
(`render_and_store_canonical_frame`), an unsigned internal rendering on the version record that
Presentation, the observations door and every later rendering of the version read
(`servable_frame_location`). Every writer reads its input in 8,192-row slices cut the same way
from any chunking (`_reslice_chunks`), so a rendering made from the stored frame has the bytes of
one made from the fresh frame. Series statistics are stored separately from the requested
rendering.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import pickle
import re
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import IO, cast
from urllib.parse import urlencode

import pyarrow as pa
import pyarrow.compute as pc
from dfo_frame_grammar import TIME_PERIOD_COLUMN, frame_sha256
from pydantic import ValidationError

from artifact_store import (
    ArtifactRecordConflictError,
    ArtifactStore,
    StoredArtifact,
    Tier,
)
from code_index.labels import PRICES_BY_QUALIFIER, family_label
from code_index.members import absent_member_reads_zero
from code_index.query import CodeIndex
from code_index.shared import get_measure_index
from code_index.types import PartyRow, level_axes, summable_axes
from dataset_store.snapshot import StoredArtifactLocation, select_servable_locations
from datasets_api.adapter import pinned_member_ref, shape_measurement_qualifier
from datasets_api.artifact_delivery import (
    FRAME_FILE_KIND,
    STATISTICS_FILE_KIND,
    artifact_download_path,
    artifact_file_name,
)
from datasets_api.capabilities import ENABLED_ENRICHMENTS
from datasets_api.codes_materialise import CapabilityUnavailable
from datasets_api.decode import build_decode_plan
from datasets_api.groupby import (
    collapse_axes,
    contributing_dcids,
    declared_axes_present,
    kept_axes,
    levels_in_frame,
    refuse_decode_gaps,
    refuse_provenance_disagreement,
    refuse_stat_var_collision,
    refuse_unknown_by_axes,
    roll_up_levels,
    sums_stat_vars,
    unsummed_axes,
    yields_grammar_frame,
)
from datasets_api.manifest_models import (
    ArtifactEntryModel,
    CitationModel,
    CitationSource,
    Manifest,
)
from datasets_api.mint_models import MintOptions
from datasets_api.output_options import movable_totals_roles
from datasets_api.party_totals import CutFrame, cut_frame, cut_party_totals, open_party_roles
from datasets_api.query_options import QueryOptionsInvalid
from datasets_api.timings import phase
from datasets_api.unit_labels import unit_currency, unit_label
from datasets_api.version_store import (
    current_rendering_keys,
    enriched_rendering_key,
    recorded_frame_sha256,
    rendering_key,
)
from datasets_api.versions import ArtifactRecord, carries_party_kind, enrichment_stamp
from dc_graph import GraphReadError
from dfo_platform import Alternative
from dfo_platform.config import PublicSiteSettings
from dfo_platform.validation import pointer_segment
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
from materialise.content_version import mint_version_from_digest
from materialise.enrichment import PARTY_KIND_FIELD, EnrichmentLookup, is_party_role
from materialise.formats import (
    DATA_SHEET_TITLE,
    FORMAT_COLUMN_CAPS,
    FRAME1_MEDIA_TYPE,
    TOTALS_SHEET_TITLE,
    Frame,
    Rendering,
    Sink,
    Table,
    WrittenArtifact,
    canonical_frame_rows,
    table_schema_for,
    write_canonical_frame,
    write_csv,
    write_excel,
    write_parquet,
)
from materialise.frame_build import ParquetFrameChunks, build_frame, unit_cell_for_rebuild
from materialise.profile import OutputProfile, PivotFeasibility
from materialise.projection import (
    LongProjection,
    PriceBasis,
    UnitNaming,
    axis_source_column,
    long_output_selection,
    period_grain,
    plan_long_projection,
    split_totals,
)
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
    known_pivot_cap_violation,
    parse_column_tokens,
    select_columns,
)
from materialise.reshape import Table as ReshapeTable

logger = logging.getLogger(__name__)

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
family ref a collapse to a frame mints when more than one member contributed
rows."""

_STATUS_COLUMN = "status"
"""`Frame.build` reserves `status`. `_full_collapse_frame` strips it before rebuilding. The local
constant avoids importing the grammar package for one name."""


@dataclass(frozen=True)
class GroupbyCollapseRendering:
    """The canonical frame from a collapse, with `stat_var` restored on every row and each kept
    code axis as a string grain column. `role_columns` names the frame's roles, each kept code
    axis's column included. It is independent of selected columns, pivot, and format.
    `frame_sha256` is the hash of `frame`. `content_version` is the version `frame` is: minted
    from that hash alone, or the base version's own token when `frame` is the base frame
    (`render_and_store`).
    `contributing_member_count` counts distinct StatVars with rows. Callers use it to warn about
    sums."""

    frame: Frame
    role_columns: tuple[str, ...]
    frame_sha256: str
    content_version: str
    stat_var_ref: str
    contributing_member_count: int


@dataclass(frozen=True)
class ReshapeResult:
    """Carries the reshaped table, warnings, and optional collapse frame. `warning_order`
    records decode-then-enrichment order for merging chunk warnings."""

    table: Table
    warnings: tuple[MaterialisationWarning, ...]
    collapse: GroupbyCollapseRendering | None
    warning_order: tuple[str, ...]


@dataclass(frozen=True)
class DerivedVersion:
    """A version a rendering mints of its own, beside the version of the frame it read: a
    `groupby` collapse's frame, the frame `without_party_totals` cut (`datasets_api.party_totals`),
    or a collapse of the cut frame. `frame_chunks` returns a fresh pass over its frame, whose
    role columns are `role_columns`. `rows` counts its rows and `distinct` the distinct values of
    each coverage dimension the cut changed, empty for a collapse of an uncut frame.
    `frame_sha256` hashes the frame and `content_version` is minted from that hash alone.
    `summed` is set when the frame is a collapse, which sums the definition's variables.
    `party_totals_left_out` holds the trace entries of the totals the cut left out
    (`PartyTotalsCut.trace_entries`), empty when none."""

    frame_chunks: Callable[[], Iterator[Frame]]
    role_columns: tuple[str, ...]
    rows: int
    distinct: Mapping[str, int]
    frame_sha256: str
    content_version: str
    summed: bool
    party_totals_left_out: tuple[dict[str, object], ...]


def check_options_supported(options: MintOptions, *, code_index: CodeIndex | None) -> None:
    """Refuses a `groupby.levels` axis outside `code_index`'s level axes (`level_axes`) with
    `QueryOptionsInvalid`, before any resolution or read. A `None` index decodes nothing, so it
    checks nothing. Frame-dependent refusals run in `_reshape`. Groupby, decode, and
    disabled-enrichment checks need the materialised frame."""
    if code_index is None or options.groupby is None or options.groupby.levels is None:
        return
    allowed = level_axes(code_index.declarations)
    for code_system in options.groupby.levels:
        if code_system not in allowed:
            raise QueryOptionsInvalid(
                pointer=f"/options/groupby/levels/{pointer_segment(code_system)}",
                detail=f"groupby.levels names {code_system!r}, which this deployment does not "
                "declare a summable hierarchical code system disjoint from its children; GET "
                f"/api/datasets/capabilities lists them as groupby_level_axes: {sorted(allowed)!r}",
            )


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
    """Cuts Arrow frames into `_RESHAPE_CHUNK_ROWS`-row tuple tables counted from the first row,
    joining the tables of short chunks, and other tables into tuple slices. A writer types its
    columns from its first chunk, so the cut keeps a rendering's bytes independent of how the
    materialiser chunked the frame. An empty frame yields one empty table, since an empty
    materialisation can yield one. `_reshape_stream` applies the bound before each streaming
    path.

    The readers downstream (decode, enrichment, groupby and the long layout) each read `rows`,
    and each `Frame.rows` access rebuilds the rows as Python objects, so every slice is converted
    here once (`canonical_frame_rows`)."""
    columns: tuple[str, ...] | None = None
    pending: list[pa.Table] = []
    pending_rows = 0
    yielded = False
    for chunk in chunks:
        if isinstance(chunk, Frame):
            columns = chunk.columns
            pending.append(chunk.table)
            pending_rows += chunk.table.num_rows
            while pending_rows >= _RESHAPE_CHUNK_ROWS:
                buffered = pa.concat_tables(pending)
                yield _frame_rows(columns, buffered.slice(0, _RESHAPE_CHUNK_ROWS))
                yielded = True
                pending = [buffered.slice(_RESHAPE_CHUNK_ROWS)]
                pending_rows -= _RESHAPE_CHUNK_ROWS
            continue
        rows = chunk.rows
        if not rows:
            yield chunk
            continue
        for start in range(0, len(rows), _RESHAPE_CHUNK_ROWS):
            yield cast(
                Table,
                ReshapeTable(columns=chunk.columns, rows=rows[start : start + _RESHAPE_CHUNK_ROWS]),
            )
    if columns is not None and (pending_rows or not yielded):
        yield _frame_rows(columns, pa.concat_tables(pending))


def _frame_rows(columns: tuple[str, ...], table: pa.Table) -> Table:
    """The canonical frame `table` as a tuple table of the rows `Frame.rows` gives."""
    return cast(Table, ReshapeTable(columns=columns, rows=tuple(canonical_frame_rows(table))))


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
    """Carries the groupby table, pending decode plan, warnings, and optional collapse
    frame."""

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
    grammar_frame: bool,
    kept: Sequence[str],
    decoded_code_systems: Sequence[str],
    code_index: CodeIndex,
    provenance_refs: Sequence[str],
    entity_columns: Sequence[str],
    subject_id: str | None,
    frame_chunks: Iterable[Table] | None,
    warnings: list[MaterialisationWarning],
) -> tuple[Table, DecodePlan | None, GroupbyCollapseRendering | None]:
    """Collapses `stat_var` then each requested axis. With `grammar_frame`, it restores
    `stat_var` on every row and builds a frame whose roles are `entity_columns` plus the
    `stat_var_{code_system}` column of each `kept` axis. That frame leaves out the column of each
    decoded axis the sum does not collapse (`groupby.unsummed_axes`), which `grammar_frame`
    holds to one value. A member with no rows counts as zero when the frame's dataset declares
    it (`absent_member_reads_zero`). Appends decode and streaming warnings to `warnings` and
    returns any unconsumed decode plan. A grammar frame requires `subject_id` unless one member
    contributed and `kept` is empty."""
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

    if not grammar_frame:
        return current, decode_plan, None

    # A sole contributor's full collapse is the raw frame unchanged, so it keeps that member's
    # dcid. A frame that keeps a code axis is new, so it names the family. Every level axis is
    # in `kept`, so an empty `kept` also means no level applied.
    if len(contributors) == 1 and not kept:
        stat_var_ref = next(iter(contributors))
    else:
        if subject_id is None:
            raise ValueError(
                "a groupby collapse to a frame that is not a sole-contributor no-op "
                "requires subject_id (the family ref is f'dcid:{subject_id}'); the "
                "sole-contributor full collapse is the only one that does not need it"
            )
        stat_var_ref = f"{_STAT_VAR_REF_PREFIX}{subject_id}"
    # A grammar frame holds each unsummed axis to one value, so its column describes the
    # family and the frame leaves it out.
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
    role_columns = (*entity_columns, *(f"{_STAT_VAR_COLUMN}_{axis}" for axis in kept))
    collapsed_frame = build_frame(role_keys=role_columns, rows=rows_as_dicts)
    current = cast(Table, collapsed_frame)
    collapsed_frame_sha256 = frame_sha256(collapsed_frame)
    collapse_info = GroupbyCollapseRendering(
        frame=collapsed_frame,
        role_columns=role_columns,
        frame_sha256=collapsed_frame_sha256,
        content_version=mint_version_from_digest(collapsed_frame_sha256),
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
    frame_chunks: Iterable[Table] | None,
) -> _GroupbyOutcome:
    """Applies `groupby` when options and `code_index` are set. Otherwise inputs pass through.
    Unknown `by` axes fail before frame classification. The collapse set is the frame's decoded,
    declared-summable axes absent from `by`. `groupby.levels` first rewrites each level axis's
    decoded labels to their depth ancestors' (`groupby.roll_up_levels`); a code above the depth
    keeps its own label.

    For a collapse or a level roll-up, decode includes resolved StatVars with zero rows. It checks
    decode gaps, StatVar collisions, and provenance agreement before summing each axis in order,
    with `stat_var` first. Each sum uses its expected population and the source's
    absent-observation rule. Naming all decoded axes in `by` with no level is a no-op. With no
    summable axis, omitting `stat_var` with multiple contributors raises
    `RollupNotSummableError`.

    `frame_chunks` streams the `stat_var` collapse over all input rows. Later axes use the reduced
    output. The streamed path preserves whole-table refusal order. When every unsummed axis holds
    one value (`groupby.yields_grammar_frame`), the sum becomes a grammar frame: `stat_var` is
    the sole contributor's dcid on a full collapse and `dcid:{subject_id}` otherwise, and each
    kept code axis is a role column. A grammar frame requires `subject_id`
    unless it is a sole contributor's full collapse.
    The frame gets a version from its own content. A sole-contributor full collapse leaves the
    frame unchanged and so shares the base version.

    Raises `RollupNotSummableError` when a family has multiple contributors and no summable axis,
    or `ValueError` when a grammar frame lacks `subject_id`.
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
    levels = levels_in_frame(
        levels=options.groupby.levels, decoded_code_systems=decoded_code_systems
    )
    if decode_plan is not None and levels:
        decode_plan = DecodePlan(
            requests=decode_plan.requests,
            lookup=roll_up_levels(decode_plan.lookup, index=code_index, levels=levels),
        )
    axes_to_collapse = collapse_axes(
        by=options.groupby.by,
        decoded_code_systems=decoded_code_systems,
        declarations=code_index.declarations,
    )
    if axes_to_collapse or levels:
        kept = kept_axes(
            by=options.groupby.by,
            decoded_code_systems=decoded_code_systems,
            declarations=code_index.declarations,
        )
        current, decode_plan, collapse_info = _reshape_groupby_collapse(
            table,
            current,
            decode_plan=decode_plan,
            axes_to_collapse=axes_to_collapse,
            grammar_frame=yields_grammar_frame(
                decoded_code_systems=decoded_code_systems,
                declarations=code_index.declarations,
                lookup=decode_plan.lookup if decode_plan is not None else {},
            ),
            kept=kept,
            decoded_code_systems=decoded_code_systems,
            code_index=code_index,
            provenance_refs=provenance_refs,
            entity_columns=entity_columns,
            subject_id=subject_id,
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
    reads: _RenderReads,
    pivot_max_output_cells: int,
    warnings: list[MaterialisationWarning],
    collapse: GroupbyCollapseRendering | None,
    code_index: CodeIndex | None,
) -> ReshapeResult:
    """Applies pending decode, `options.columns`, and `options.pivot`. Real and decoded columns take
    precedence over enrichment tokens. Enabled `{entity}_{field}` tokens read their values through
    `reads` (`_RenderReads.fields_of`): a party's kind from the code index, the rest in one
    batched graph read against the render's deadline, whose failure leaves cells empty.
    Recognised disabled fields raise `CapabilityUnavailable`, and unknown tokens raise
    `ColumnUnknownError` with alternatives. Decode, selection, then pivot run in that order. A token
    list with no table or decoded column does not select, so every default column stays.

    Without a pivot the output is long and `long_output_selection` widens the selection: decoded
    axis columns come with `stat_var` unless the list names some of them. Long output leaves out
    `status` and writes each role's `<role>_label` by default (`materialise.projection`), so both
    tokens are accepted and ignored there, and a selection naming `<role>_label` keeps `<role>`.
    A frame-layout request (`FRAME_LAYOUT`) is not long output: its selection is the list as
    parsed, as for pivot output.

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
    long_output = options.pivot is None and options.layout is None

    parsed: ParsedColumns | None = None
    named_roles: tuple[str, ...] = ()
    if options.columns is not None:
        tokens: Sequence[str] = options.columns
        if long_output:
            tokens, named_roles = _long_output_tokens(
                options.columns, table_columns=post_decode_columns, entity_columns=entity_columns
            )
        try:
            parsed = parse_column_tokens(
                tokens,
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
    if select is not None:
        select = (*select, *(role for role in named_roles if role not in select))
    if long_output:
        select = long_output_selection(
            select, axis_columns=_axis_columns(post_decode_columns, code_index=code_index)
        )
    enrichment_plan: EnrichmentPlan | None = None
    if parsed is not None and parsed.enrichments:
        refs = _enrichment_refs(current, enrichments=parsed.enrichments)
        enrichment_plan = EnrichmentPlan(
            requests=parsed.enrichments,
            lookup=reads.fields_of(refs, fields={field for _, field in parsed.enrichments}),
            # A pivot's index and columns group rows, so a kind that varies with the period
            # would split one party into several index rows; its kind cell stays year-free.
            parties=None if pivot is not None else reads.enrichment.parties,
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
    reads: _RenderReads,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None = None,
    resolved_stat_var_refs: Sequence[str],
    provenance_refs: Sequence[str],
    subject_id: str | None = None,
    frame_chunks: Iterable[Table] | None = None,
) -> ReshapeResult:
    """Applies decode, groupby, selection, enrichment, and pivot before rendering. Inert transforms
    leave the input table unchanged.

    A supplied `code_index` enables decode-plan construction even when no decode option is set.
    `None` skips decoding.
    `resolved_stat_var_refs` supplies zero-row members to groupby population checks, and
    `provenance_refs` govern its source-wide missing-as-zero rule. `subject_id` is used only for
    the collapse frame's `stat_var`. `frame_chunks` streams the first
    collapse axis. `None` processes `table` in memory.

    Raises `ColumnUnknownError` for unknown columns, `CapabilityUnavailable` for disabled
    enrichment, `RollupNotSummableError` when a family has no summable axis, and `ValueError` when
    a collapse to a frame lacks `subject_id`.
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
        frame_chunks=frame_chunks,
    )
    return _reshape_select_pivot_enrich(
        groupby_outcome.table,
        decode_plan=groupby_outcome.decode_plan,
        options=options,
        entity_columns=entity_columns,
        reads=reads,
        pivot_max_output_cells=pivot_max_output_cells,
        warnings=list(groupby_outcome.warnings),
        collapse=groupby_outcome.collapse,
        code_index=code_index,
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


def stored_artifact_file_name(
    location: StoredArtifactLocation,
    *,
    citation: CitationModel,
    prepared_at: datetime,
    content_version: str,
) -> str:
    """Name a linked artifact from its version's cited source names, `prepared_at` and
    `content_version`, as the mint names it.

    A definition's sources are its citation's. A ref the citation lacks stands as its name.
    """
    refs = [source.ref for source in citation.sources]
    return artifact_file_name(
        source_names=cited_source_names(refs, citation.sources_by_ref()),
        prepared_at=prepared_at,
        content_version=content_version,
        media_type=location.media_type,
        rendering=location.rendering,
        kind=_file_kind(location.rendering),
        shape=_file_shape(location.rendering),
    )


def linked_under(manifest: Manifest, subject_id: str) -> Manifest:
    """`manifest` with every artifact's download link naming `subject_id`, the id the request
    named. A click on the link then checks the revocations the request honoured: a request
    naming the served id honours revocations under it, and a request naming a posted id
    honours those under the posted and the served id (`doors.snapshot.read_under_served_id`).
    """
    return manifest.model_copy(
        update={
            "artifacts": [
                artifact.model_copy(
                    update={
                        "url": artifact_download_path(
                            subject_id=subject_id,
                            content_version=manifest.content_version,
                            artifact_digest=artifact.artifact_digest,
                        )
                    }
                )
                for artifact in manifest.artifacts
            ]
        }
    )


def _file_kind(rendering: Mapping[str, object]) -> str | None:
    """The file-name kind of a linked artifact: the statistics file's or the canonical frame's,
    else `None` for a requested rendering."""
    if is_statistics_rendering(rendering):
        return STATISTICS_FILE_KIND
    if is_canonical_frame_rendering(rendering):
        return FRAME_FILE_KIND
    return None


# A text value names its field in a file name only when it is a short lowercase word.
_SHAPE_WORD = re.compile(r"[a-z][a-z0-9]{0,15}")


def _file_shape(rendering: Mapping[str, object]) -> str | None:
    """The file-name shape token of a requested rendering, `None` for the default long table
    and for a file with a `_file_kind`.

    The shaping fields are the rendering without the keys every requested rendering carries
    whatever its shape (`format`, `bundle`, a bundle's `RENDITION_KEY` and a bare file's
    `LAYOUT_KEY = DATA_LAYOUT_VERSION`), so the default table keeps its name. The token names
    each field in key order, by its value when that is a short lowercase word (`frame`, `incomplete`) and by its
    key otherwise (`columns`, `groupby`, `pivot`), then the first six hex characters of the
    fields' `rendering_key`, so two renderings that differ in any field, such as two pivots on
    other axes, take two names: `pivot-1a2b3c`, `frame-4d5e6f`."""
    if _file_kind(rendering) is not None:
        return None
    shaping = {
        key: value
        for key, value in rendering.items()
        if key not in ("format", "bundle", RENDITION_KEY)
        and not (key == LAYOUT_KEY and value == DATA_LAYOUT_VERSION)
    }
    if not shaping:
        return None
    words = [
        value if isinstance(value, str) and _SHAPE_WORD.fullmatch(value) else key
        for key, value in sorted(shaping.items())
    ]
    return "-".join([*words, rendering_key(shaping)[:6]])


def stored_artifact_entry(
    location: StoredArtifactLocation,
    *,
    subject_id: str,
    content_version: str,
    enrichment_digest: str | None,
) -> ArtifactEntryModel:
    """The manifest entry of one linked artifact location, carrying its stable download link.
    Building it signs nothing.

    `enrichment_digest` is the serving code index's (`CodeIndex.enrichment_digest`). The entry
    carries it as `enrichment_stamp` when the location is linked under the key it salts
    (`version_store.enriched_rendering_key`), which only a file holding party-kind cells read
    under it is (`versions.link_digest`). A file linked under any other key states no
    stamp: the key does not say which index an older file was rendered under."""
    stamped = (
        enrichment_digest is not None
        and carries_party_kind(location.rendering)
        and location.rendering_key == enriched_rendering_key(location.rendering, enrichment_digest)
    )
    return ArtifactEntryModel(
        artifact_digest=location.artifact_digest,
        rendering=dict(location.rendering),
        bytes=location.size_bytes,
        media_type=location.media_type,
        # The manifest uses first-sighting time. Retention uses this tier's `stored_at`, and
        # pinning does not reset creation time.
        created_at=location.created_at.isoformat(),
        tier=location.tier,
        url=artifact_download_path(
            subject_id=subject_id,
            content_version=content_version,
            artifact_digest=location.artifact_digest,
        ),
        enrichment_stamp=enrichment_digest if stamped else None,
    )


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
    code_index: CodeIndex | None,
    description: str | None,
) -> MemberProvenance:
    """One bundle member, titled by its sources and years, with each column documented from
    `scan`."""
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
        description=description,
    )


@dataclass(frozen=True)
class _TotalsFile:
    """A bundle's second CSV data file, holding the total rows (`_TotalRows`): its path, the
    written file, its scan and the first table written, which types its schema."""

    path: str
    written: WrittenArtifact
    scan: TableScan
    frame: Table


def _totals_path(data_path: str) -> str:
    """The totals file beside `data_path`: `data/<slug>-totals.<format>`."""
    stem, extension = data_path.rsplit(".", 1)
    return f"{stem}-totals.{extension}"


def _write_data(
    chunks: Iterable[Table],
    *,
    options: MintOptions,
    rendering: Rendering,
    sink: Sink,
    totals: Callable[[], Iterable[Table]] | None,
) -> WrittenArtifact:
    """Writes one data file in `options.format`. An XLSX file writes `totals`, called once its
    data sheet is written, on its second sheet (`formats.write_excel`). The other writers take
    no totals."""
    if options.format == "xlsx":
        return write_excel(chunks, rendering=rendering, sink=sink, totals=totals)
    return _FORMAT_WRITERS[options.format](chunks, rendering=rendering, sink=sink)


def _write_totals_entry(
    zf: zipfile.ZipFile,
    totals: _TotalRows,
    *,
    data_path: str,
    rendering: Rendering,
    scan: TableScan,
) -> _TotalsFile | None:
    """Writes the total rows as the CSV entry `_totals_path(data_path)` of `zf`, scanned by
    `scan`. `None`, writing nothing, when no row is a total."""
    if totals.first is None:
        return None
    path = _totals_path(data_path)
    with zf.open(
        zipfile.ZipInfo(path, date_time=_ZIP_FIXED_DATE_TIME), mode="w", force_zip64=True
    ) as entry:
        written = write_csv(scan.observe(totals.tables()), rendering=rendering, sink=entry)
    return _TotalsFile(path=path, written=written, scan=scan, frame=totals.first)


_OBSERVATION_DESCRIPTION = "One row per observation."


def _members(
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
    totals: _TotalRows | None,
    totals_file: _TotalsFile | None,
) -> tuple[MemberProvenance, ...]:
    """The bundle's members: the data file, then the totals file when the total rows are
    written as one (`_write_totals_entry`). Each description says where the total rows are,
    and the data file's says when it holds no row because every row is a total. An XLSX file
    holds both sheets, so when its data sheet holds no row the totals sheet's first table
    types its schema."""
    if (
        options.format == "xlsx"
        and not frame.rows
        and totals is not None
        and totals.first is not None
    ):
        frame = totals.first
    main = _member_provenance(
        spec_id=spec_id,
        written=written,
        frame=frame,
        rendering=rendering,
        data_path=data_path,
        sources=sources,
        scan=scan,
        code_index=code_index,
        description=_data_description(
            options,
            scan=scan,
            totals=totals,
            totals_path=totals_file.path if totals_file is not None else None,
        ),
    )
    if totals_file is None:
        return (main,)
    return (
        main,
        _member_provenance(
            spec_id=spec_id,
            written=totals_file.written,
            frame=totals_file.frame,
            rendering=rendering,
            data_path=totals_file.path,
            sources=sources,
            scan=totals_file.scan,
            code_index=code_index,
            description=(
                "One row per observation of a total, a party that holds other parties' "
                "figures, such as a regional total. Adding these rows to those of "
                f"{data_path} counts those figures twice."
            ),
        ),
    )


def _data_description(
    options: MintOptions, *, scan: TableScan, totals: _TotalRows | None, totals_path: str | None
) -> str | None:
    """What one row of the data file holds, `None` for a pivot, which spreads observations
    across columns. With total rows set apart, it names where they are: `totals_path` for a
    CSV, the second sheet for an XLSX, whose `scan` covers both sheets."""
    if options.pivot is not None:
        return None
    if totals is None or not totals.rows:
        return _OBSERVATION_DESCRIPTION
    if totals_path is not None:
        if not scan.rows:
            return f"No rows: every observation is of a total, and the totals are in {totals_path}."
        return (
            "One row per observation of a party or a residual. The rows of totals are in "
            f"{totals_path}."
        )
    if scan.rows == totals.rows:
        return (
            f"No rows on the sheet {DATA_SHEET_TITLE}: every observation is of a total, and the "
            f"totals are on the sheet {TOTALS_SHEET_TITLE}."
        )
    return (
        f"One row per observation of a party or a residual, on the sheet {DATA_SHEET_TITLE}. "
        f"The rows of totals are on the sheet {TOTALS_SHEET_TITLE}."
    )


def _table_scan(
    role_columns: Sequence[str], *, options: MintOptions, code_index: CodeIndex | None
) -> TableScan:
    """A scan of one data file written under `options`, told the pivot that shaped it and the
    axis code columns long output writes, each code taking its code system's plain name as its
    kind."""
    pivot = options.pivot
    return TableScan(
        role_columns=role_columns,
        axis_kinds=(
            {name: declaration.plain_name for name, declaration in code_index.declarations.items()}
            if code_index is not None and pivot is None
            else None
        ),
        pivot_index=pivot.index if pivot is not None else None,
        pivot_columns=pivot.columns if pivot is not None else None,
    )


def _codes(
    scans: Sequence[TableScan],
    *,
    citations: Mapping[str, CitationSource],
    family: tuple[str, str] | None,
    code_index: CodeIndex | None,
    reads: _RenderReads,
) -> BundleCodes:
    """The codes table of the scanned files. Cited sources take their cited names, a collapsed
    family's `(ref, label)` takes its label, and an axis code takes its `code_index` label, the
    name its `<code_system>_label` cell holds. The graph names every other code through the
    render's `reads`, so the read shares the render's deadline and makes none after a failed
    name read."""
    known = {ref: cited.label for ref, cited in citations.items()}
    if family is not None:
        known[family[0]] = family[1]
    if code_index is not None:
        for scan in scans:
            for column in scan.axis_kinds:
                for code in scan.codes(column):
                    if (row := code_index.code_by_dcid(code)) is not None:
                        known[code] = row.label
    return bundle_codes(scans, known_labels=known, lookup=reads.lookup)


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
    summarised from `scans`, and linked under the public site. The website link (when
    `site.site_dataset_links` is on) goes only to a definition with a described identity: the
    website replays the identity through Search's `dataset_resubmit`, which refuses the pinned
    arm (`kind: "stat_var"`) that `statvar_set` mints."""
    subject_id = str(spec.get("spec_id"))
    names = _source_names([source for member in members for source in member.sources])
    origin = site.public_site_url
    site_link = (
        f"{origin}/data?{urlencode({'spec': subject_id, 'v': content_version})}"
        if site.site_dataset_links and _site_replayable(spec)
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
        rebuild_request=_rebuild_request(spec, options),
    )


def _store_artifact(
    *,
    data: bytes,
    media_type: str,
    rendering: Mapping[str, object],
    subject_id: str,
    content_version: str,
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
    )


def _commit_stored(
    stored: StoredArtifact,
    *,
    media_type: str,
    rendering: Mapping[str, object],
    subject_id: str,
    content_version: str,
) -> ArtifactEntryModel:
    """Builds the `ArtifactEntryModel` of a committed `put` or `open_put` result, carrying its
    stable download link. Streaming writes pass the committed sink result directly, avoiding a
    whole-artifact buffer."""
    return ArtifactEntryModel(
        artifact_digest=stored.digest,
        rendering=dict(rendering),
        bytes=stored.size_bytes,
        media_type=media_type,
        created_at=datetime.now(UTC).isoformat(),
        tier="working",
        url=artifact_download_path(
            subject_id=subject_id,
            content_version=content_version,
            artifact_digest=stored.digest,
        ),
    )


def _long_output_tokens(
    columns: Sequence[str], *, table_columns: Sequence[str], entity_columns: Sequence[str]
) -> tuple[list[str], tuple[str, ...]]:
    """The `options.columns` tokens a long output parses, and the roles whose name or kind the
    list asked for. Long output writes `status` nowhere, and writes each role's `<role>_label`
    and each party role's `<role>_party_kind` by default (`materialise.projection`), so those
    tokens are dropped. A list that selects keeps each named role's entity column, which its
    name and kind sit beside."""
    defaults = {
        f"{role}_label": role for role in entity_columns if f"{role}_label" not in table_columns
    } | {
        f"{role}_{PARTY_KIND_FIELD}": role
        for role in entity_columns
        if is_party_role(role) and f"{role}_{PARTY_KIND_FIELD}" not in table_columns
    }
    tokens = [token for token in columns if token != _STATUS_COLUMN and token not in defaults]
    named_roles = tuple(dict.fromkeys(defaults[t] for t in columns if t in defaults))
    return tokens, named_roles


def _axis_columns(columns: Sequence[str], *, code_index: CodeIndex | None) -> tuple[str, ...]:
    """The decoded `stat_var_<code_system>` columns among `columns`, in code-system order."""
    if code_index is None:
        return ()
    return tuple(
        column
        for code_system in sorted(code_index.declarations)
        if (column := axis_source_column(code_system)) in columns
    )


def _axis_codes(code_index: CodeIndex | None) -> dict[str, dict[str, str | None]]:
    """Each declared code system's `{label: code dcid}`, the lookup a long output's axis code
    columns read (`materialise.projection.plan_long_projection`). A label two codes of one system
    share maps to `None`, so its rows get an empty code cell and an `enrichment_unavailable`
    warning naming the code column."""
    if code_index is None:
        return {}
    codes: dict[str, dict[str, str | None]] = {
        code_system: {} for code_system in code_index.declarations
    }
    for row in code_index.codes:
        by_label = codes.get(row.code_system)
        if by_label is None:
            continue
        by_label[row.label] = None if row.label in by_label else row.dcid
    return codes


def _native_price_basis(
    code_index: CodeIndex | None,
    *,
    resolved_stat_var_refs: Sequence[str],
    shape_qualifier: str | None,
) -> Callable[[object], PriceBasis | None]:
    """The price basis of a native currency unit on a row, from the row's `stat_var`: the
    `measurementQualifier` the code index records for that StatVar (`PRICES_BY_QUALIFIER`).
    A summed family ref takes the basis every resolved StatVar shares, when the index knows
    them all and they agree. Any other row takes `shape_qualifier`, the identity shape's
    `measurementQualifier`, which every member shares. The code index holds the CRS StatVars
    only, so a series from another source gets its basis from the shape."""
    shape_basis = PRICES_BY_QUALIFIER.get(shape_qualifier) if shape_qualifier else None
    if code_index is None:
        return lambda _stat_var: shape_basis
    by_bare = {ref.removeprefix(_STAT_VAR_REF_PREFIX): ref for ref in resolved_stat_var_refs}
    by_ref: dict[object, PriceBasis | None] = {
        by_bare[row.dcid]: PRICES_BY_QUALIFIER.get(row.measurement_qualifier or "")
        for row in code_index.statvars
        if row.dcid in by_bare
    }
    bases = set(by_ref.values())
    shared = next(iter(bases)) if len(bases) == 1 and len(by_ref) == len(by_bare) else shape_basis
    return lambda stat_var: by_ref.get(stat_var, shared)


@dataclass
class _RenderReads:
    """Where a render's names and enrichment fields come from, shared by the reshape
    enrichment, the long layout and the bundle's codes table. `known` names a cell value
    outright: a cited source's label, a summed family's label. Every other value is read
    through `enrichment` by `lookup` against one absolute `deadline`, the render's. After one
    read fails, `lookup` refuses without reading, so a graph outage costs the render at most
    one read's wait."""

    enrichment: EnrichmentLookup
    known: Mapping[str, str]
    deadline: float | None
    failed: bool = False

    @classmethod
    def for_render(
        cls,
        *,
        enrichment: EnrichmentLookup,
        citations: Mapping[str, CitationSource],
        family: tuple[str, str] | None,
        deadline: float | None,
    ) -> _RenderReads:
        known = {ref: cited.label for ref, cited in citations.items()}
        if family is not None:
            known[family[0]] = family[1]
        return cls(
            enrichment=enrichment,
            known=known,
            deadline=deadline,
        )

    def lookup(self, nodes: Sequence[str]) -> Mapping[str, Mapping[str, str]]:
        """`enrichment`'s fields of `nodes`. Raises `GraphReadError` when this read fails or
        an earlier one of the render did."""
        if self.failed:
            raise GraphReadError("an earlier name read of this render failed")
        try:
            return self.enrichment.lookup(nodes, deadline=self.deadline)
        except GraphReadError:
            self.failed = True
            raise

    def fields_of(
        self, nodes: Sequence[str], *, fields: Collection[str]
    ) -> dict[str, dict[str, str]]:
        """Each of `nodes` mapped to those of its enrichment `fields` that could be read. A
        party's `party_kind` is the one the code index declares
        (`EnrichmentLookup.party_kind`), read with no graph call; a party the index does not
        list has none. One `lookup` reads the graph fields of every node, and runs only when one
        is asked for. A read that fails, or follows a failed one, leaves those fields out. An
        absent field's cells stay empty and count as `enrichment_unavailable`."""
        read: dict[str, dict[str, str]] = {node: {} for node in nodes}
        if PARTY_KIND_FIELD in fields:
            for node, node_fields in read.items():
                if kind := self.enrichment.party_kind(node):
                    node_fields[PARTY_KIND_FIELD] = kind
        graph_fields = set(fields) - {PARTY_KIND_FIELD}
        if not graph_fields or not nodes or self.failed:
            return read
        try:
            looked_up = self.lookup(nodes)
        except GraphReadError:
            logger.warning(
                "download graph read failed; %d values keep empty enrichment cells",
                len(nodes),
                exc_info=True,
            )
            return read
        for node in nodes:
            for field_name, value in looked_up.get(node, {}).items():
                if field_name in graph_fields and value:
                    read[node][field_name] = value
        return read


_TOTALS_SPOOL_MEMORY_BYTES = 16 * 1024 * 1024
"""Total rows held in memory before `_TotalRows` moves them to a temporary file."""

_TOTALS_FORMATS = frozenset(("csv", "xlsx"))
"""The formats whose long output keeps total rows out of its main table."""


class _TotalRows:
    """The total rows a long CSV or XLSX render keeps out of its main table
    (`materialise.projection.split_totals`): the rows where one of `roles` holds a party of
    kind `total`. Its cell reads `total`, or `whole_total` for a published whole-population
    total inside its years. The cells `party`, `private` and `residual` stay in the main table.
    `roles` are the roles that can hold a total
    (`datasets_api.output_options.movable_totals_roles`), the ones the options document's
    `long_layout_totals` names, read from the definition before the first chunk, since the
    layout streams.

    With `keep` set the rows are spooled, one pickled table per chunk, in memory up to
    `_TOTALS_SPOOL_MEMORY_BYTES` and in a temporary file beyond it, so they wait out the main
    table without holding a large share of the data in memory. A bare CSV has no place for
    them and keeps none. `rows` counts the rows set apart either way, and `first` holds the
    first non-empty table kept, which types the totals file's schema."""

    def __init__(self, *, roles: Sequence[str], keep: bool) -> None:
        self.roles = tuple(roles)
        self.rows = 0
        self.first: Table | None = None
        # The spool outlives this call: `render_and_store` closes it once the bundle is written.
        self._spool = (
            tempfile.SpooledTemporaryFile(max_size=_TOTALS_SPOOL_MEMORY_BYTES)  # noqa: SIM115
            if keep
            else None
        )

    @classmethod
    def for_render(
        cls,
        options: MintOptions,
        *,
        identity: object,
        entity_columns: Sequence[str],
        parties: Mapping[str, PartyRow],
        datasets: Collection[str],
    ) -> _TotalRows | None:
        """The total rows of a render under `options`, or `None` for a format outside
        `_TOTALS_FORMATS`, a pivot, or a definition with no role that can hold a total, read
        over the code index's `parties` and the provenances the definition read, `datasets`."""
        if options.format not in _TOTALS_FORMATS or options.pivot is not None:
            return None
        if not isinstance(identity, Mapping):
            return None
        roles = movable_totals_roles(
            cast("Mapping[str, object]", identity),
            entity_columns,
            parties=parties,
            datasets=datasets,
        )
        if not roles:
            return None
        return cls(roles=roles, keep=options.bundle or options.format == "xlsx")

    def add(self, table: Table) -> None:
        if not table.rows:
            return
        self.rows += len(table.rows)
        if self._spool is None:
            return
        if self.first is None:
            self.first = table
        pickle.dump(table, self._spool, protocol=pickle.HIGHEST_PROTOCOL)

    def tables(self) -> Iterator[Table]:
        """The kept tables in the order they were added."""
        if self._spool is None:
            return
        self._spool.seek(0)
        while True:
            try:
                yield pickle.load(self._spool)
            except EOFError:
                return

    def close(self) -> None:
        if self._spool is not None:
            self._spool.close()


def _nonempty_chunks(chunks: Iterable[Table]) -> Iterator[Table]:
    """`chunks` less its empty tables, or its last table when every one is empty, so a
    writer's first chunk, which types its columns, holds rows whenever any chunk does. Total
    rows leave a long output's chunks, which can empty one."""
    last: Table | None = None
    yielded = False
    for chunk in chunks:
        if chunk.rows:
            yielded = True
            yield chunk
        else:
            last = chunk
    if not yielded and last is not None:
        yield last


def _frame_period_grain(frame_chunks: Callable[[], Iterator[Table]]) -> str | None:
    """The period grain of the whole frame `frame_chunks` holds
    (`materialise.projection.period_grain`), each distinct `time_period` read once
    (`pc.unique`). A frame held as Parquet reads that column alone of each row group
    (`ParquetFrameChunks.column_chunks`), an Arrow frame its table, and any other table row by
    row."""

    def periods() -> Iterator[object]:
        if isinstance(frame_chunks, ParquetFrameChunks):
            for column in frame_chunks.column_chunks(TIME_PERIOD_COLUMN):
                yield from pc.unique(column).to_pylist()
            return
        for chunk in frame_chunks():
            if TIME_PERIOD_COLUMN not in chunk.columns:
                continue
            if isinstance(chunk, Frame):
                yield from pc.unique(chunk.table[TIME_PERIOD_COLUMN]).to_pylist()
                continue
            at = chunk.columns.index(TIME_PERIOD_COLUMN)
            yield from (row[at] for row in chunk.rows)

    return period_grain(periods())


@dataclass
class _LongLayout:
    """Puts long output in its published layout (`materialise.projection`) at the writer
    boundary. The layout is planned from the first table it sees, and every later chunk of the
    same render has the same columns.

    A party's kind cell is the one the code index declares (`materialise.enrichment.party_kind_cell`)
    for the year of its row, read with no graph call, so the split of totals is the same in every
    chunk and every render of a version, whether or not a graph read succeeds. A whole total
    (`PartyRow.whole`) reads `whole_total` in the dataset and years its marker holds and
    `total` outside them. A party the index does not list has an
    empty kind cell and stays in the main table. Each chunk's names come from one
    `reads.fields_of` read before it is projected, the read the reshape enrichment also uses,
    and a value is asked for once per layout. A graph read that fails or runs out of time
    leaves the names it would have given empty and stops further reads for the whole render
    (`_RenderReads`), so the render completes with those cells empty. `warnings` then counts
    the empty name, kind and code cells per column as `enrichment_unavailable`.

    `grain` is the whole frame's period grain (`materialise.projection.period_grain`), read by
    `render_and_store` before the first chunk. With `year` every `time_period` cell is an
    integer, and otherwise every cell stays text. With `totals` set, `apply` returns the rows of
    parties and residuals and passes the total rows to `totals`.

    A frame-layout request (`FRAME_LAYOUT`) gets a layout with `frame` set, whose `apply`
    returns each table unchanged and reads no names, so its file holds the frame's grammar
    columns."""

    entity_columns: Sequence[str]
    axis_codes: Mapping[str, Mapping[str, str | None]]
    unit_naming: UnitNaming
    reads: _RenderReads
    grain: str | None = None
    frame: bool = False
    totals: _TotalRows | None = None
    projection: LongProjection | None = None
    names: dict[str, str] = field(default_factory=dict)
    asked: set[str] = field(default_factory=set)
    unavailable: Counter[str] = field(default_factory=Counter)

    @classmethod
    def for_render(
        cls,
        *,
        options: MintOptions,
        entity_columns: Sequence[str],
        code_index: CodeIndex | None,
        resolved_stat_var_refs: Sequence[str],
        shape_qualifier: str | None,
        reads: _RenderReads,
        grain: str | None,
        totals: _TotalRows | None,
    ) -> _LongLayout:
        if options.layout == FRAME_LAYOUT:
            return cls(
                entity_columns=entity_columns,
                axis_codes={},
                unit_naming=UnitNaming(
                    currency_of=unit_currency,
                    native_price_basis=lambda _stat_var: None,
                    label=unit_label,
                ),
                reads=reads,
                frame=True,
            )
        return cls(
            entity_columns=entity_columns,
            axis_codes=_axis_codes(code_index),
            unit_naming=UnitNaming(
                currency_of=unit_currency,
                native_price_basis=_native_price_basis(
                    code_index,
                    resolved_stat_var_refs=resolved_stat_var_refs,
                    shape_qualifier=shape_qualifier,
                ),
                label=unit_label,
            ),
            reads=reads,
            grain=grain,
            totals=totals,
        )

    def apply(self, table: Table) -> Table:
        if self.frame:
            return table
        if self.projection is None:
            self.projection = plan_long_projection(
                table.columns,
                entity_columns=self.entity_columns,
                axis_codes=self.axis_codes,
                unit_naming=self.unit_naming,
                grain=self.grain,
            )
        self._read_names(self.projection.refs_to_name(table))
        projected = self.projection.apply(
            table, names=self.names, parties=self.reads.enrichment.parties
        )
        self.unavailable.update(projected.unavailable)
        if self.totals is None:
            return cast(Table, projected.table)
        parties, totals = split_totals(projected.table, roles=self.totals.roles)
        self.totals.add(cast(Table, totals))
        return cast(Table, parties)

    def _read_names(self, refs: Sequence[str]) -> None:
        new = [ref for ref in refs if ref not in self.asked]
        self.asked.update(new)
        to_read: list[str] = []
        for ref in new:
            if ref in self.reads.known:
                self.names[ref] = self.reads.known[ref]
            else:
                to_read.append(ref)
        if not to_read:
            return
        # StatVar cells carry the frame's `dcid:` spelling. The graph names the bare dcid.
        bare = {ref: ref.removeprefix(_STAT_VAR_REF_PREFIX) for ref in to_read}
        fields = self.reads.fields_of(tuple(dict.fromkeys(bare.values())), fields=("label",))
        for ref, node in bare.items():
            if name := fields[node].get("label"):
                self.names[ref] = name

    def names_complete(self) -> bool:
        """Whether every name read of the render so far answered."""
        return not self.reads.failed

    def warnings(
        self, warnings: Sequence[MaterialisationWarning]
    ) -> tuple[MaterialisationWarning, ...]:
        """`warnings` with each `enrichment_unavailable` column named as the output names it,
        then one `enrichment_unavailable` per name or code column with empty cells, in column
        order."""
        renamed = self.projection.renamed if self.projection is not None else {}
        out: list[MaterialisationWarning] = []
        for warning in warnings:
            if warning["code"] == "enrichment_unavailable":
                entry = cast(EnrichmentUnavailableWarning, warning)
                if entry["column"] in renamed:
                    warning = EnrichmentUnavailableWarning(
                        code="enrichment_unavailable",
                        severity=entry["severity"],
                        message=entry["message"],
                        column=renamed[entry["column"]],
                        count=entry["count"],
                    )
            out.append(warning)
        columns = self.projection.columns if self.projection is not None else ()
        out.extend(
            enrichment_unavailable_warning(column=column, count=self.unavailable[column])
            for column in columns
            if self.unavailable[column]
        )
        return tuple(out)


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


def _stored_pivot_facts(
    output_profile: OutputProfile | None, plan: PivotPlan
) -> PivotFeasibility | None:
    """The stored profile's facts for the plan's `(index, columns)` pair, when it has them."""
    if output_profile is None:
        return None
    return next(
        (
            facts
            for facts in output_profile.pivots
            if (facts.index, facts.columns) == (plan.index, plan.columns)
        ),
        None,
    )


def refuse_pivot_over_stored_caps(
    options: MintOptions, *, output_profile: OutputProfile | None, pivot_max_output_cells: int
) -> None:
    """Raise the column-cap or cell-cap refusal that a bare pivot of the frame `output_profile`
    describes raises before its scan (`fold_pivot_chunks` with the stored facts).

    The caller applies this only where no refusal outranks the caps before the fold: a pivot
    with no `groupby` (whose collapse decision needs the frame) and no `columns` (whose parse
    refuses first).
    """
    assert options.pivot is not None, "refuse_pivot_over_stored_caps requires options.pivot"
    plan = PivotPlan(
        index=options.pivot.index,
        columns=options.pivot.columns,
        column_cap=_column_cap(options.format),
        cell_cap=pivot_max_output_cells,
    )
    violation = known_pivot_cap_violation(
        _stored_pivot_facts(output_profile, plan),
        column_cap=plan.column_cap,
        cell_cap=plan.cell_cap,
    )
    if violation is not None:
        raise violation


def _stream_bare_pivot(
    frame_chunks: Iterable[Table],
    *,
    options: MintOptions,
    entity_columns: Sequence[str],
    reads: _RenderReads,
    pivot_max_output_cells: int,
    code_index: CodeIndex | None,
    resolved_stat_var_refs: Sequence[str],
    provenance_refs: Sequence[str],
    subject_id: str,
    output_profile: OutputProfile | None = None,
) -> ReshapeResult:
    """Folds pivot output from chunked rows in one pass. Decode and selection run per chunk, and
    the output stays within `column_cap` and `cell_cap`.

    A groupby that collapses nothing still validates `by`. When `stat_var` is omitted, multiple
    contributors require a decoded summable axis. This refusal runs before pivot, as in whole-table
    processing. `distinct_dcids` gathers the required refs during the fold.

    The fold counts rows per enrichment ref, and one `_RenderReads.fields_of` read after it
    gives the whole-table missing-value warnings. Pivot output carries the index and pivoted
    values.

    `output_profile` describes the frame these chunks carry. Its facts for the requested pair let
    the fold refuse a cap before scanning and name the `varying` columns of a non-unique pair. A
    request whose `no_summable_axis` refusal can apply ignores them, since that refusal reads every
    chunk.
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

    # The refusal needs every contributing StatVar, so a fold that reads one chunk and stops on a
    # stored cap fact would decide it on a prefix. Such a request reads the whole stream.
    rollup_unsummable = (
        options.groupby is not None
        and code_index is not None
        and _STAT_VAR_COLUMN not in options.groupby.by
        and not declared_axes_present(
            decoded_code_systems=tuple(
                code_system for _entity_column, code_system in decode_plan.requests
            )
            if decode_plan is not None
            else (),
            declarations=code_index.declarations,
        )
    )

    try:
        outcome = fold_pivot_chunks(
            _decoded_selected_chunks(),
            index=pivot_plan.index,
            columns=pivot_plan.columns,
            column_cap=pivot_plan.column_cap,
            cell_cap=pivot_plan.cell_cap,
            classification=classification,
            value_column=pivot_plan.value_column,
            known=None if rollup_unsummable else _stored_pivot_facts(output_profile, pivot_plan),
        )
    except ColumnUnknownError as exc:
        exc.pointer = "/options/pivot"
        raise

    if rollup_unsummable and len(distinct_dcids) > 1:
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
        lookup = reads.fields_of(all_refs, fields={field for _, field in parsed.enrichments})
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
    Callers run it inside `open_put` so that refusal aborts the artifact. `names_complete`,
    read after the writer drains, is `False` when a long output's name read failed
    (`_LongLayout`), so the caller links the artifact as incomplete."""

    chunks_for_writer: Iterable[Table]
    collapse: GroupbyCollapseRendering | None
    finish: Callable[[], tuple[Table, tuple[MaterialisationWarning, ...]]]
    names_complete: Callable[[], bool] = lambda: True


def _reshape_stream_plain(
    frame_chunks: Iterable[Table],
    *,
    options: MintOptions,
    entity_columns: Sequence[str],
    pivot_max_output_cells: int,
    code_index: CodeIndex | None,
    resolved_stat_var_refs: Sequence[str],
    provenance_refs: Sequence[str],
    shape_qualifier: str | None,
    name_reads: _RenderReads,
    grain: str | None,
    totals: _TotalRows | None,
) -> _StreamedRender:
    """Streams row-wise decode, selection, and enrichment by reshaping each chunk and putting it
    in the long layout before passing it to the writer. `_merge_enrichment_warnings` combines
    per-chunk counts."""
    layout = _LongLayout.for_render(
        options=options,
        entity_columns=entity_columns,
        code_index=code_index,
        resolved_stat_var_refs=resolved_stat_var_refs,
        shape_qualifier=shape_qualifier,
        reads=name_reads,
        grain=grain,
        totals=totals,
    )
    collected_warnings: list[MaterialisationWarning] = []
    first_reshaped_chunk: list[Table] = []
    first_warning_order: list[tuple[str, ...]] = []

    def _projected_chunks() -> Iterator[Table]:
        for raw_chunk in frame_chunks:
            chunk_result = _reshape(
                raw_chunk,
                options=options,
                entity_columns=entity_columns,
                reads=name_reads,
                pivot_max_output_cells=pivot_max_output_cells,
                code_index=code_index,
                resolved_stat_var_refs=resolved_stat_var_refs,
                provenance_refs=provenance_refs,
            )
            collected_warnings.extend(chunk_result.warnings)
            if not first_warning_order:
                first_warning_order.append(chunk_result.warning_order)
            yield layout.apply(chunk_result.table)

    def _reshaped_chunks() -> Iterator[Table]:
        for chunk in _nonempty_chunks(_projected_chunks()):
            if not first_reshaped_chunk:
                first_reshaped_chunk.append(chunk)
            yield chunk

    def _finish() -> tuple[Table, tuple[MaterialisationWarning, ...]]:
        assert first_reshaped_chunk, (
            "render_and_store requires frame_chunks to yield at least one chunk"
        )
        return (
            first_reshaped_chunk[0],
            layout.warnings(
                _merge_enrichment_warnings(collected_warnings, order=first_warning_order[0])
            ),
        )

    return _StreamedRender(
        chunks_for_writer=_reshaped_chunks(),
        collapse=None,
        finish=_finish,
        names_complete=layout.names_complete,
    )


def _reshape_stream(
    frame_chunks: Iterable[Table],
    *,
    options: MintOptions,
    entity_columns: Sequence[str],
    pivot_max_output_cells: int,
    code_index: CodeIndex | None,
    resolved_stat_var_refs: Sequence[str],
    provenance_refs: Sequence[str],
    subject_id: str,
    shape_qualifier: str | None,
    name_reads: _RenderReads,
    grain: str | None,
    totals: _TotalRows | None,
    output_profile: OutputProfile | None = None,
) -> _StreamedRender:
    """Reslices chunks, then dispatches the frame through one of four paths:

    - Plain decode, selection, and enrichment stream each reshaped chunk to the writer.
    - A collapsing groupby folds `stat_var` across chunks, then writes the reduced table.
    - A bare pivot folds output within its width and cell caps.
    - A no-op groupby streams per chunk and checks its refusal after the writer drains.

    Groupby and pivot folds bound output size. Other paths retain chunk-sized row data.
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
            pivot_max_output_cells=pivot_max_output_cells,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            provenance_refs=provenance_refs,
            shape_qualifier=shape_qualifier,
            name_reads=name_reads,
            grain=grain,
            totals=totals,
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
        will_stream_groupby_collapse = sums_stat_vars(
            by=options.groupby.by,
            levels=options.groupby.levels,
            decoded_code_systems=probe_decoded_code_systems,
            declarations=code_index.declarations,
        )

    if will_stream_groupby_collapse and first_chunk is not None:
        # `_reshape` recomputes this decision from the same inputs. A chained pivot runs on the
        # reduced output groups.
        result = _reshape(
            first_chunk,
            options=options,
            entity_columns=entity_columns,
            reads=name_reads,
            pivot_max_output_cells=pivot_max_output_cells,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            provenance_refs=provenance_refs,
            subject_id=subject_id,
            frame_chunks=rest_of_frame_chunks,
        )
        if options.pivot is not None:
            return _StreamedRender(
                chunks_for_writer=(result.table,),
                collapse=result.collapse,
                finish=lambda: (result.table, result.warnings),
            )
        # The collapse frame keeps its grammar columns. Only the written table is projected.
        layout = _LongLayout.for_render(
            options=options,
            entity_columns=entity_columns,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            shape_qualifier=shape_qualifier,
            reads=name_reads,
            grain=grain,
            totals=totals,
        )
        collapsed = layout.apply(result.table)
        collapsed_warnings = layout.warnings(result.warnings)
        return _StreamedRender(
            chunks_for_writer=(collapsed,),
            collapse=result.collapse,
            finish=lambda: (collapsed, collapsed_warnings),
            names_complete=layout.names_complete,
        )

    if first_chunk is not None and options.pivot is not None and not will_stream_groupby_collapse:
        # Stream bare pivots and pivots after a no-op groupby. The fold checks that groupby families
        # with multiple contributors include a summable axis.
        result = _stream_bare_pivot(
            rest_of_frame_chunks,
            options=options,
            entity_columns=entity_columns,
            reads=name_reads,
            pivot_max_output_cells=pivot_max_output_cells,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            provenance_refs=provenance_refs,
            subject_id=subject_id,
            output_profile=output_profile,
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
    layout = _LongLayout.for_render(
        options=options,
        entity_columns=entity_columns,
        code_index=code_index,
        resolved_stat_var_refs=resolved_stat_var_refs,
        shape_qualifier=shape_qualifier,
        reads=name_reads,
        grain=grain,
        totals=totals,
    )
    collected_warnings: list[MaterialisationWarning] = []
    first_reshaped_chunk: list[Table] = []
    first_warning_order: list[tuple[str, ...]] = []
    distinct_dcids: set[str] = set()
    stat_var_pos: list[int] = []

    def _projected_chunks() -> Iterator[Table]:
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
                reads=name_reads,
                pivot_max_output_cells=pivot_max_output_cells,
                code_index=code_index,
                resolved_stat_var_refs=resolved_stat_var_refs,
                provenance_refs=provenance_refs,
            )
            collected_warnings.extend(chunk_result.warnings)
            if not first_warning_order:
                first_warning_order.append(chunk_result.warning_order)
            yield layout.apply(chunk_result.table)

    def _groupby_passthrough_chunks() -> Iterator[Table]:
        for chunk in _nonempty_chunks(_projected_chunks()):
            if not first_reshaped_chunk:
                first_reshaped_chunk.append(chunk)
            yield chunk

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
        return first_reshaped_chunk[0], layout.warnings(merged)

    return _StreamedRender(
        chunks_for_writer=_groupby_passthrough_chunks(),
        collapse=None,
        finish=_finish,
        names_complete=layout.names_complete,
    )


@dataclass(frozen=True)
class RenderedDefinition:
    """The `render_and_store` result. `collapse` is set for a groupby collapse to a frame.
    `derived` is set when the artifact belongs to a version of its own (`DerivedVersion`), and
    `datasets_api.serve` and `datasets_api.current` use it to write that version's canonical
    frame, statistics and version record.
    `labels_complete` is `False` when a bundle's code label read
    (`BundleCodes.label_read_complete`) or a long output's name read failed, and the caller
    then links the artifact as incomplete (`incomplete_rendering`)."""

    artifact: ArtifactEntryModel
    warnings: tuple[MaterialisationWarning, ...]
    collapse: GroupbyCollapseRendering | None
    labels_complete: bool = True
    derived: DerivedVersion | None = None


def render_and_store(
    *,
    frame_chunks: Callable[[], Iterator[Table]],
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
    output_profile: OutputProfile | None = None,
    deadline: float | None = None,
    base_frame_sha256: str | None = None,
) -> RenderedDefinition:
    """Renders one definition to a stored artifact and returns its reshape warnings.
    `frame_chunks` returns a fresh pass over the materialised frame's chunks in order. Long
    output reads the frame's period grain first (`_frame_period_grain`), which types every
    `time_period` cell of the file alike, and every render reads one pass to write.

    A bundle is built from the version's facts: `citations` (the version's
    `CitationModel.sources_by_ref()`, with empty documentation and licence fields filled by
    `datasets_api.manifest.healed_citation`) names and documents each source, and `prepared_at`
    is the version's first-mint instant. Its data file is `data/<sources slug>.<format>`, its
    columns are documented and its codes listed from the rows the writer streams, and the codes
    are named by one batched graph read through `enrichment`. Every graph name read of the
    render shares `deadline`, and a failed read stops the rest (`_RenderReads`). A fresh render of one
    version therefore matches the artifact linked for it while the sources' documentation and
    the graph's code names are unchanged. `site` supplies the origin of every link.

    Decode, selection, and enrichment stream one chunk at a time. Enrichment warnings merge into
    one count per column. Groupby folds across chunks. Pivot folds to bounded output. No-op groupby
    also streams, so wide results can exceed 1,000,000 rows. The raw-fetch guard allows 1.5 times
    the coverage estimate.

    `code_index` controls decoding, and the artifact carries its `enrichment_digest` as
    `enrichment_stamp` when it holds party-kind cells (`versions.enrichment_stamp`, over
    the role columns of the frame it renders). `resolved_stat_var_refs` and `provenance_refs`
    support groupby population and source checks. A collapse frame mints its own version,
    which its artifact uses; every other artifact uses `content_version`. `serve` writes
    version records.

    A collapse frame that is the base frame is the base version, so its collapse carries
    `content_version`. The frames are compared by hash. `base_frame_sha256` is the hash of
    the frame `frame_chunks` holds (`stored_frame_sha256`), given when that frame is read back
    from the store, whose version may carry a token of an earlier frame grammar. Without it
    `content_version` is the token minted from `frame_chunks`, which equals the collapse's
    exactly when the frames are equal.
    The family ref a summed collapse writes in `stat_var` is labelled with
    `code_index.labels.family_label` over `spec`'s identity.

    `options.without_party_totals` cuts the group totals of the open party roles of `spec`'s
    identity out of the frame before any reshape (`datasets_api.party_totals`), reading the
    kinds from `enrichment.parties`. A cut that leaves rows out renders from the cut frame,
    which is a version of its own, so the artifact and any collapse of it belong to a version
    minted from their own frame (`RenderedDefinition.derived`). A cut that leaves nothing out
    renders the frame as it is. Every chunk `frame_chunks` yields is then a `Frame`.
    """
    check_options_supported(options, code_index=code_index)
    subject_id = str(spec.get("spec_id"))
    identity = spec.get("identity")
    # A cut reads `Frame` chunks, which `datasets_api.current` and `datasets_api.serve` pass.
    frames = cast("Callable[[], Iterator[Frame]]", frame_chunks)
    party_cut = (
        cut_party_totals(
            frames,
            roles=open_party_roles(identity, entity_columns),
            parties=enrichment.parties,
        )
        if options.without_party_totals
        else None
    )
    cut = (
        cut_frame(frames, party_cut, role_columns=entity_columns)
        if party_cut is not None and party_cut.left_out
        else None
    )
    if cut is not None:
        frame_chunks = cut.frame_chunks
    subject_label = (
        family_label(
            cast("Mapping[str, object]", identity),
            index=code_index,
            measure_index=get_measure_index(),
        )
        if code_index is not None and isinstance(identity, Mapping)
        else None
    )
    name_reads = _RenderReads.for_render(
        enrichment=enrichment,
        citations=citations,
        family=(
            (f"{_STAT_VAR_REF_PREFIX}{subject_id}", subject_label)
            if subject_label is not None
            else None
        ),
        deadline=deadline,
    )
    totals = _TotalRows.for_render(
        options,
        identity=identity,
        entity_columns=entity_columns,
        parties=enrichment.parties,
        datasets=provenance_refs,
    )
    with phase("render"):
        rendering = Rendering()
        long_output = options.pivot is None and options.layout is None
        grain = _frame_period_grain(frame_chunks) if long_output else None
        streamed = _reshape_stream(
            frame_chunks(),
            options=options,
            entity_columns=entity_columns,
            pivot_max_output_cells=pivot_max_output_cells,
            code_index=code_index,
            resolved_stat_var_refs=resolved_stat_var_refs,
            provenance_refs=provenance_refs,
            subject_id=subject_id,
            shape_qualifier=(
                shape_measurement_qualifier(cast("Mapping[str, object]", identity))
                if isinstance(identity, Mapping)
                else None
            ),
            name_reads=name_reads,
            grain=grain,
            totals=totals,
            output_profile=output_profile,
        )
        chunks_for_writer = streamed.chunks_for_writer
        collapse = streamed.collapse
        if collapse is not None and collapse.frame_sha256 == base_frame_sha256:
            collapse = replace(collapse, content_version=content_version)
        derived = _derived_version(
            collapse, cut=cut, content_version=content_version, entity_columns=entity_columns
        )
        effective_content_version = (
            derived.content_version if derived is not None else content_version
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
                        scan = _table_scan(entity_columns, options=options, code_index=code_index)
                        with zf.open(
                            zipfile.ZipInfo(data_path, date_time=_ZIP_FIXED_DATE_TIME),
                            mode="w",
                            force_zip64=True,
                        ) as entry:
                            written = _write_data(
                                scan.observe(chunks_for_writer),
                                options=options,
                                rendering=rendering,
                                sink=entry,
                                totals=(
                                    (lambda: scan.observe(totals.tables()))
                                    if totals is not None
                                    else None
                                ),
                            )
                        # `streamed.finish()` runs while `sink`/`zf` are
                        # still open, so a refusal it raises (the deferred
                        # `no_summable_axis` check) still aborts this write
                        # -- see `_StreamedRender`'s own docstring.
                        table, warnings = streamed.finish()
                        totals_file = (
                            _write_totals_entry(
                                zf,
                                totals,
                                data_path=data_path,
                                rendering=rendering,
                                scan=_table_scan(
                                    entity_columns, options=options, code_index=code_index
                                ),
                            )
                            if totals is not None and options.format == "csv"
                            else None
                        )
                        scans = (scan,) if totals_file is None else (scan, totals_file.scan)
                        bundle_codes_read = _codes(
                            scans,
                            citations=citations,
                            family=_family(
                                collapse, subject_id=subject_id, subject_label=subject_label
                            ),
                            code_index=code_index,
                            reads=name_reads,
                        )
                        labels_complete = (
                            bundle_codes_read.label_read_complete and streamed.names_complete()
                        )
                        codes = bundle_codes_read.entries
                        members = _members(
                            spec_id=subject_id,
                            written=written,
                            frame=table,
                            rendering=rendering,
                            data_path=data_path,
                            sources=sources,
                            scan=scan,
                            options=options,
                            code_index=code_index,
                            totals=totals,
                            totals_file=totals_file,
                        )
                        files = assemble_bundle(
                            spec=spec,
                            members=members,
                            reproducing_call=reproducing_call,
                            context=_bundle_context(
                                spec=spec,
                                content_version=effective_content_version,
                                members=members,
                                scans=scans,
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
                    written = _write_data(
                        chunks_for_writer,
                        options=options,
                        rendering=rendering,
                        sink=sink,
                        totals=totals.tables if totals is not None else None,
                    )
                    table, warnings = streamed.finish()
                    media_type = written.media_type
                    labels_complete = streamed.names_complete()
            stored = sink.result()
        except ArtifactRecordConflictError as exc:
            # See `_store_artifact`'s own identical clause: this call sites
            # the same conflict context (`subject_id`/`content_version`)
            # onto the exception before it propagates.
            exc.subject_id = subject_id
            exc.content_version = effective_content_version
            raise
        finally:
            if totals is not None:
                totals.close()

    artifact = _commit_stored(
        stored,
        media_type=media_type,
        rendering=artifact_rendering(options, site=site),
        subject_id=subject_id,
        content_version=effective_content_version,
    )
    stamp = enrichment_stamp(
        artifact.rendering,
        role_columns=derived.role_columns if derived is not None else entity_columns,
        enrichment_digest=code_index.enrichment_digest if code_index is not None else None,
    )
    return RenderedDefinition(
        artifact=artifact.model_copy(update={"enrichment_stamp": stamp}),
        warnings=warnings,
        collapse=collapse,
        labels_complete=labels_complete,
        derived=derived,
    )


def _derived_version(
    collapse: GroupbyCollapseRendering | None,
    *,
    cut: CutFrame | None,
    content_version: str,
    entity_columns: Sequence[str],
) -> DerivedVersion | None:
    """The version a render's artifact belongs to when it is not `content_version`, the version
    of the frame the render read: the collapse when it is a frame of its own, else the cut
    frame, else `None`. A collapse equal to the frame it summed (one contributor, nothing
    kept) is that frame, so a collapse of the cut frame equal to it is the cut frame."""
    if collapse is not None and collapse.content_version == content_version:
        return None
    cut_version = mint_version_from_digest(cut.frame_sha256) if cut is not None else None
    left_out = cut.left_out if cut is not None else ()
    if collapse is not None and collapse.content_version != cut_version:
        frame = collapse.frame
        return DerivedVersion(
            frame_chunks=lambda: iter((frame,)),
            role_columns=collapse.role_columns,
            rows=frame.num_rows,
            distinct={},
            frame_sha256=collapse.frame_sha256,
            content_version=collapse.content_version,
            summed=True,
            party_totals_left_out=left_out,
        )
    if cut is None or cut_version is None:
        return None
    return DerivedVersion(
        frame_chunks=cut.frame_chunks,
        role_columns=tuple(entity_columns),
        rows=cut.rows,
        distinct=cut.distinct,
        frame_sha256=cut.frame_sha256,
        content_version=cut_version,
        summed=False,
        party_totals_left_out=left_out,
    )


BUNDLE_LAYOUT_VERSION = 3
"""The bundle layout `rendition_token` hashes. Bump it with any change to a bundle's files or
their contents for unchanged data and settings (a file added, renamed or reworded, a column or
code described differently), so every version renders and links a new bundle on its next
request while each old bundle stays listed under its own rendering key. Version 3 writes
integer years and keeps total rows out of a CSV or XLSX data table (`_TotalRows`)."""

_WITHOUT_PARTY_TOTALS_KEY = "without_party_totals"
"""The rendering key `options.without_party_totals` is recorded under, present only when true."""

RENDITION_KEY = "rendition"
"""The bundle rendering's rendition token (`rendition_token`). A bare rendering has none."""

LAYOUT_KEY = "layout"
"""The bare rendering's data layout (`DATA_LAYOUT_VERSION`). A bundle has none, since its
rendition token covers the layout of the data file inside it."""

METADATA_KEY = "metadata"
"""Marks the rendering of an artifact a graph metadata read failed for
(`incomplete_rendering`). No other rendering carries it."""

METADATA_INCOMPLETE = "incomplete"
"""The `METADATA_KEY` value of an artifact rendered without its sources' documentation or
without graph names, because the read failed."""

DATA_LAYOUT_VERSION = 3
"""The column layout of a bare data file. Bump it with any change to a bare file's columns or
cells for unchanged data and options, so a request renders a new file under a new rendering key
and cannot match one written in the old layout. Version 2 is the long layout of
`materialise.projection`. Version 3 adds integer years and keeps total rows out of the main
table (`_TotalRows`): a bundle's CSV writes them to a second file, a bare CSV drops them, and an
XLSX moves them to a second sheet. A rendering without `LAYOUT_KEY` predates
version 2."""

FRAME_LAYOUT = "frame"
"""The `options.layout` value that asks for a bare parquet file in the frame layout: the
frame's grammar columns as reshaped (`stat_var`, the roles, decoded `stat_var_<code_system>`
axes, `time_period`, `value`, `unit` with a converted unit as its JCS request text,
`measurementMethod`, `provenance`, `status`), the file machine readers parse. Its rendering
carries `LAYOUT_KEY = FRAME_LAYOUT` from the request and no `DATA_LAYOUT_VERSION` stamp. A
change to this layout ships under a new value."""


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
    `bundle` always appears and an unset reshape never does) and without a false
    `without_party_totals`, so a rendering that leaves no totals out has the key of the same
    options with no such field, plus `RENDITION_KEY` for a bundle
    and `LAYOUT_KEY = DATA_LAYOUT_VERSION` for a bare file in the long layout. A frame-layout
    request's `options` already carry `LAYOUT_KEY = FRAME_LAYOUT`, so its rendering is the
    options dump as sent. The rendering key hashes this dict
    (`version_store.rendering_key`), so a layout or settings change gives new artifacts a new
    key and leaves old links matching only themselves.
    """
    rendering = options.model_dump(mode="json", exclude_none=True)
    if not options.without_party_totals:
        del rendering[_WITHOUT_PARTY_TOTALS_KEY]
    if options.bundle:
        rendering[RENDITION_KEY] = rendition_token(site)
    elif options.layout is None:
        rendering[LAYOUT_KEY] = DATA_LAYOUT_VERSION
    return rendering


def incomplete_rendering(rendering: Mapping[str, object]) -> dict[str, object]:
    """The `rendering` an artifact is stored and linked under when a graph metadata read of
    its render failed: `rendering` plus `METADATA_KEY`.

    Its rendering key differs from `rendering`'s, so the artifact is linked and downloadable
    while no request's key matches it (`artifact_rendering_keys`), and the next render of
    `rendering` reads the graph again and links its artifact beside this one.
    """
    return {**rendering, METADATA_KEY: METADATA_INCOMPLETE}


def is_incomplete_rendering(rendering: Mapping[str, object]) -> bool:
    """Whether `rendering` is one `incomplete_rendering` made."""
    return rendering.get(METADATA_KEY) == METADATA_INCOMPLETE


def complete_rendering(rendering: Mapping[str, object]) -> dict[str, object]:
    """`rendering` without `METADATA_KEY`: the rendering a request for the artifact matches."""
    return {key: value for key, value in rendering.items() if key != METADATA_KEY}


def artifact_rendering_keys(
    options: MintOptions, *, site: PublicSiteSettings, enrichment_digest: str | None
) -> tuple[str, ...]:
    """The rendering keys a current artifact rendered under `options` can be linked under: the
    keys (`version_store.current_rendering_keys`) of `artifact_rendering(options, site=site)`.
    The definition door's currency check, the bundle-link check and the operations poll all
    derive a request's keys here, so they narrow a version to the same rendering.

    `enrichment_digest` is the serving code index's (`CodeIndex.enrichment_digest`). A
    rendering that asks for a party-kind cell has two keys, salted with the digest and with
    `versions.NO_PARTY_ROLE`, and its file is linked under the one its frame's roles pick;
    any other rendering has one. The salt stays out of `artifact_rendering`, so the rendering
    a caller selects an artifact by is the same under every digest.
    """
    return current_rendering_keys(artifact_rendering(options, site=site), enrichment_digest)


def current_enrichment_locations(
    locations: Sequence[StoredArtifactLocation], *, enrichment_digest: str
) -> list[StoredArtifactLocation]:
    """`locations` without the files that a re-render under `enrichment_digest` has replaced.

    After a change to a party's kind a version holds both generations of a file with party-kind
    cells, with equal `rendering` dicts, and a caller selecting by `rendering` takes the first
    it meets. A file is hidden when it is linked under no current key
    (`version_store.current_rendering_keys`) and a file of the same `rendering` is linked under
    one. That is the only case: a stale file with no current twin stays, so an old layout, a
    superseded or pre-rendition bundle, an incomplete file and a pinned file linked before the
    digest existed are listed unchanged. A hidden link stays downloadable by its digest.
    """
    current = [
        location.rendering_key in current_rendering_keys(location.rendering, enrichment_digest)
        for location in locations
    ]
    # A rendering identifies itself by its unsalted key: `enriched_rendering_key` with no digest.
    renderings_under_current = {
        enriched_rendering_key(location.rendering, None)
        for location, is_current in zip(locations, current, strict=True)
        if is_current
    }
    return [
        location
        for location, is_current in zip(locations, current, strict=True)
        if is_current
        or enriched_rendering_key(location.rendering, None) not in renderings_under_current
    ]


def rendering_options(
    rendering: Mapping[str, object], *, site: PublicSiteSettings
) -> MintOptions | None:
    """The `MintOptions` this build renders `rendering` under, or `None` when no options do.

    `rendering` round-trips through `artifact_rendering`, so `None` covers the stored frame
    (canonical or frame1) and statistics renderings, a bundle of a superseded rendition, a
    bare file in an old layout and an incomplete rendering (`incomplete_rendering`), whose
    `METADATA_KEY` no options produce. The options of an incomplete rendering's complete
    form (`complete_rendering`) render the artifact its request named.
    """
    fields = dict(rendering)
    fields.pop(RENDITION_KEY, None)
    if fields.get(LAYOUT_KEY) == DATA_LAYOUT_VERSION:
        del fields[LAYOUT_KEY]
    try:
        options = MintOptions.model_validate(fields)
    except ValidationError:
        return None
    return options if artifact_rendering(options, site=site) == dict(rendering) else None


CANONICAL_FRAME_RENDERING: Mapping[str, object] = {"format": "frame"}
"""Fixed rendering key for the canonical frame, independent of request options. Mints with
different options therefore reuse the same `version_artifact` row for one frame's bytes."""

_FRAME1_RENDERING: Mapping[str, object] = {"format": "frame1"}
"""The rendering of the frame1 document a version stored before canonical frames holds."""


def is_canonical_frame_rendering(rendering: Mapping[str, object]) -> bool:
    """Matches the canonical frame rendering, which every version's canonical frame carries."""
    return dict(rendering) == CANONICAL_FRAME_RENDERING


def is_frame1_rendering(rendering: Mapping[str, object]) -> bool:
    """Matches the frame1 rendering of a version's stored frame1 document."""
    return dict(rendering) == _FRAME1_RENDERING


def stored_frame_location(
    locations: Sequence[StoredArtifactLocation],
) -> StoredArtifactLocation | None:
    """The version's stored frame among `locations`: its canonical frame, else its frame1
    document, else `None`. Read the bytes with `materialise.formats.read_stored_frame` and the
    location's media type."""
    canonical = next(
        (location for location in locations if is_canonical_frame_rendering(location.rendering)),
        None,
    )
    if canonical is not None:
        return canonical
    return next(
        (location for location in locations if location.media_type == FRAME1_MEDIA_TYPE), None
    )


def stored_frame_sha256(
    location: StoredArtifactLocation,
    table: Table,
    *,
    recorded: str | None,
    locations: Sequence[StoredArtifactLocation] = (),
) -> str:
    """The frame hash (`dfo_frame_grammar.frame_sha256`) of the stored frame at `location`,
    which `table` holds as `materialise.formats.read_stored_frame` read it: `recorded`, the
    hash on the version's record (`StoredVersion.frame_sha256`), when the record holds one.
    Otherwise a frame1 document is the frame's canonical bytes, so the digest of one at
    `location` or among `locations`, the version's links the caller holds, is that hash.
    A canonical frame with neither is hashed."""
    held = recorded_frame_sha256(recorded, (location, *locations))
    return held if held is not None else frame_sha256(cast(Frame, table))


def servable_frame_location(
    locations: Sequence[StoredArtifactLocation], *, now: datetime
) -> StoredArtifactLocation | None:
    """The servable location of the version's stored frame (`stored_frame_location`).
    Servability follows `select_servable_locations`, so a pinned location wins over a working
    one."""
    return stored_frame_location(select_servable_locations(locations, now=now))


STATISTICS_MEDIA_TYPE = "application/vnd.apache.parquet"
"""Both `series-statistics/1` and requested Parquet use `application/vnd.apache.parquet`. Rendering
metadata identifies statistics with `{"statistics": "series"}`."""

_STATISTICS_RENDERING_KEY = "statistics"
_STATISTICS_RENDERING_VALUE = "series"


_STATISTICS_RENDERING: Mapping[str, object] = {
    _STATISTICS_RENDERING_KEY: _STATISTICS_RENDERING_VALUE
}
"""Fixed rendering key for series statistics, independent of request options."""


def is_statistics_rendering(rendering: Mapping[str, object]) -> bool:
    """Matches the `series-statistics/1` rendering key. Snapshot retrieval uses this to include
    statistics independently of the requested format."""
    return rendering.get(_STATISTICS_RENDERING_KEY) == _STATISTICS_RENDERING_VALUE


def render_and_store_series_statistics(
    *,
    data: bytes,
    subject_id: str,
    content_version: str,
    artifact_store: ArtifactStore,
) -> ArtifactEntryModel:
    """Stores each materialisation's `series-statistics/1` Parquet as a listable artifact,
    independent of the requested format. Manifests list it when `series_count` is set."""
    return _store_artifact(
        data=data,
        media_type=STATISTICS_MEDIA_TYPE,
        rendering=_STATISTICS_RENDERING,
        subject_id=subject_id,
        content_version=content_version,
        artifact_store=artifact_store,
    )


def render_and_store_canonical_frame(
    *,
    frame_chunks: Iterable[Frame],
    subject_id: str,
    content_version: str,
    artifact_store: ArtifactStore,
) -> ArtifactRecord:
    """Streams the canonical frame of `frame_chunks` into the store as an unsigned artifact,
    holding at most one row group (`materialise.formats.write_canonical_frame`), whatever the
    frame's size. Readers find it on the version record by `CANONICAL_FRAME_RENDERING`. Encoding
    runs inside the shared `store` phase, so when it runs beside the requested write that phase
    sums both threads' time."""
    with phase("store"):
        try:
            with artifact_store.open_put(tier=Tier.working) as sink:
                written = write_canonical_frame(frame_chunks, sink=sink)
            stored = sink.result()
        except ArtifactRecordConflictError as exc:
            exc.subject_id = subject_id
            exc.content_version = content_version
            raise
    return ArtifactRecord(
        artifact_digest=stored.digest,
        rendering=dict(CANONICAL_FRAME_RENDERING),
        media_type=written.media_type,
    )
