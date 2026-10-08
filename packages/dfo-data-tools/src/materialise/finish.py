"""Builds versions, warnings, statistics, profiles, and traces after generation returns its
revision snapshot. `_finish_member` handles single members and `_finish_group` handles combined
materialisations. Previews reuse member minting and warnings without a trace."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dc_sdmx import ContainedIn, EntityFilter

# `pipeline` imports this module during initialisation. `MaterialisationResult` resolves when
# `_finish_member` runs, after `pipeline` has loaded.
from materialise import pipeline
from materialise.codes import MATERIALISE_CODE_CATALOGUE, MaterialisationWarning
from materialise.content_version import (
    FRAME_GRAMMAR_VERSION,
    ConversionPreimage,
    MintPreimage,
    mint_version,
    mint_version_from_digest,
    native_unit_source_revision,
    recipe_body_source_revision,
    source_revision_preimage,
)
from materialise.convert import ConversionRecord
from materialise.enrichment import EnrichmentLookup
from materialise.errors import MaterialisationUsageError
from materialise.facets import FacetSelection, binds_no_place
from materialise.frame_build import Frame

# `_build_warnings` checks outcome classes at runtime. `generation` imports only the
# `MaterialisationRequest` annotation under `TYPE_CHECKING`, so these imports create no cycle.
from materialise.generation import (
    _bare_stat_var_ref,
    _GroupOutcome,
    _MemberOutcome,
    _PreviewOutcome,
)
from materialise.profile import compute_output_profile, compute_output_profile_from_chunks
from materialise.revision import RevisionSnapshot
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace, ResolvedStatVar
from materialise.window import coverage_below_scope, unserved_entities

if TYPE_CHECKING:
    from materialise.pipeline import MaterialisationRequest

# Each conversion trace entry caps its members' merged examples at the per-member limit, 10.
_MAX_TRACE_EXAMPLES = 10


def _capped_distinct_examples[ExampleT: (str, tuple[str, str])](
    examples: Iterable[ExampleT],
) -> tuple[list[ExampleT], bool]:
    """Deduplicates and sorts examples, caps them at `_MAX_TRACE_EXAMPLES`, and reports
    whether distinct examples were cut. Truncation counts examples, not affected rows."""
    distinct = sorted(set(examples))
    return distinct[:_MAX_TRACE_EXAMPLES], len(distinct) > _MAX_TRACE_EXAMPLES


def _conversion_trace_entry(
    tables: ConversionPreimage, members: Sequence[tuple[str, ConversionRecord]]
) -> dict[str, object]:
    """Builds one `ManifestTrace.conversion` entry for the members converted under one recipe
    and table-version pair. `members` lists each distinct (bare StatVar ref,
    `ConversionRecord.native_unit_ref`) pair once, sorted, so records over the same StatVar
    and native unit name it once. Fallback and projected counts sum over every
    record, because each record counts its own definition's rows; their examples are
    deduplicated, sorted, and capped at `_MAX_TRACE_EXAMPLES`, with `truncated` reflecting
    distinct examples, not row counts. `deflator_table_version` is omitted when no deflator
    table joined. Fallback pairs are JSON lists."""
    entry: dict[str, object] = {
        "recipe": tables.recipe_id,
        "rate_table_version": tables.exchange_rate_table_version,
    }
    if tables.deflator_table_version is not None:
        entry["deflator_table_version"] = tables.deflator_table_version
    entry["members"] = [
        {"stat_var": stat_var, "native_unit": native_unit}
        for stat_var, native_unit in sorted(
            {(_bare_stat_var_ref(stat_var), record.native_unit_ref) for stat_var, record in members}
        )
    ]

    capped_fallback, fallback_truncated = _capped_distinct_examples(
        pair for _, record in members for pair in record.fallback_examples
    )
    entry["fallback"] = {
        "count": sum(record.fallback_count for _, record in members),
        "examples": [list(pair) for pair in capped_fallback],
        "truncated": fallback_truncated,
    }

    capped_projected, projected_truncated = _capped_distinct_examples(
        example for _, record in members for example in record.projected_examples
    )
    entry["projected"] = {
        "count": sum(record.projected_count for _, record in members),
        "examples": list(capped_projected),
        "truncated": projected_truncated,
    }
    return entry


def _conversion_trace(
    converted: Iterable[tuple[str, ConversionRecord]],
) -> tuple[dict[str, object], ...] | None:
    """Builds `ManifestTrace.conversion`: one entry per distinct (recipe, rate-table version,
    deflator-table version), naming the members converted under it, or `None` when no member
    converted. Entries sort by recipe then table versions, and members by StatVar, so the
    trace is the same whatever order members arrive in. A group mixing recipes or table
    versions gets one entry for each."""
    by_tables: dict[ConversionPreimage, list[tuple[str, ConversionRecord]]] = {}
    for stat_var, record in sorted(converted, key=lambda pair: pair[0]):
        tables = ConversionPreimage.from_conversion_record(record)
        assert tables is not None  # a record always projects to a preimage
        by_tables.setdefault(tables, []).append((stat_var, record))
    if not by_tables:
        return None
    return tuple(
        _conversion_trace_entry(tables, by_tables[tables])
        for tables in sorted(
            by_tables,
            key=lambda t: (
                t.recipe_id,
                t.exchange_rate_table_version,
                t.deflator_table_version is not None,
                t.deflator_table_version or "",
            ),
        )
    )


def _recipe_body_revisions(conversion: ConversionPreimage) -> tuple[dict[str, object], ...]:
    """The converted frame's `recipe_body/<recipe>` revision as a zero- or one-entry tuple, for
    its trace's `source_revisions`. The token hashes this revision, so the trace must carry it
    for a client to recompute the token through the grammar alone."""
    entry = recipe_body_source_revision(conversion)
    return () if entry is None else (entry,)


@dataclass(frozen=True)
class _MemberVersionMint:
    """One member's version and preimage inputs. Trace fields reuse these same computed
    revisions and facets, including a converted member's native-unit revision."""

    content_version: str
    graph_revision: str
    source_revisions: tuple[dict[str, object], ...]
    facets: tuple[dict[str, object], ...]
    """The grammar-shaped preimage facets `content_version` was actually
    minted from (`FacetSelection.as_grammar_facet`'s own closed
    `{axis, ref, basis}` shape) -- distinct from `ManifestTrace.facets`
    (`as_trace_entry`'s own wider, 7-key wire shape). `_finish_member`
    carries this onto `pipeline.MaterialisationResult.mint_preimage`.
    """


def _mint_member_content_version(
    request: MaterialisationRequest,
    selections: Mapping[str, FacetSelection],
    conversion_record: ConversionRecord | None,
    frame: Frame,
    *,
    revision_snapshot: RevisionSnapshot,
) -> _MemberVersionMint:
    """Mints the version and preimage shared by member and preview results. The preview
    supplies its full survivor frame. A member supplies its delivered frame.

    Converted members add the native-unit and recipe-body revisions, so the trace holds every
    revision the token hashes. Axes with `ref=None` stay in the trace but are omitted from the
    grammar preimage, which rejects null refs. The frame hash distinguishes different content."""
    normalised = revision_snapshot.normalised()
    source_revisions: tuple[dict[str, object], ...] = tuple(
        source_revision_preimage(source=sr.source, revision=sr.revision)
        for sr in normalised.source_revisions
    )
    conversion = ConversionPreimage.from_conversion_record(conversion_record)
    if conversion is not None:
        native_unit = selections["unit"].ref
        assert native_unit is not None  # a converted member always resolves a native unit
        source_revisions = (
            *source_revisions,
            native_unit_source_revision(
                stat_var=_bare_stat_var_ref(request.stat_var), native_unit=native_unit
            ),
            *_recipe_body_revisions(conversion),
        )
    facets = tuple(
        selection.as_grammar_facet()
        for selection in selections.values()
        if selection.ref is not None
    )

    content_version = mint_version(
        frame,
        graph_revision=normalised.graph_revision,
        source_revisions=source_revisions,
        facets=facets,
        conversion=conversion,
    )
    return _MemberVersionMint(
        content_version=content_version,
        graph_revision=normalised.graph_revision,
        source_revisions=source_revisions,
        facets=facets,
    )


def _finish_member(
    request: MaterialisationRequest,
    outcome: _MemberOutcome,
    *,
    revision_snapshot: RevisionSnapshot,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
) -> pipeline.MaterialisationResult:
    """After generation returns the revision snapshot, builds the version, warnings,
    per-series statistics, output profile, and trace. `output_profile_seconds` records the
    profile computation time."""
    mint = _mint_member_content_version(
        request,
        outcome.selections,
        outcome.conversion_record,
        outcome.frame,
        revision_snapshot=revision_snapshot,
    )
    content_version = mint.content_version
    normalised_graph_revision = mint.graph_revision
    source_revisions = mint.source_revisions
    # Statistics derive from the frame but do not enter its content-version preimage.
    statistics_sink = SeriesStatisticsSink()
    statistics_sink.write(compute_series_statistics_table(outcome.frame))
    series_statistics = statistics_sink.finish()
    profile_start = time.perf_counter()
    output_profile = compute_output_profile(
        outcome.frame, role_columns=request.declared_roles, enrichment=enrichment
    )
    output_profile_seconds = time.perf_counter() - profile_start

    trace = ManifestTrace(
        applied_window=(
            outcome.applied_window.as_trace_entry() if outcome.applied_window is not None else None
        ),
        graph_revision=normalised_graph_revision,
        source_revisions=source_revisions,
        resolved_stat_vars=(
            ResolvedStatVar(
                ref=request.stat_var,
                roles=request.declared_roles,
                observation_count=outcome.coverage.rows,
            ).as_trace_entry(),
        ),
        facets=tuple(selection.as_trace_entry() for selection in outcome.selections.values()),
        normalisations=(),
        conversion=(
            _conversion_trace(((request.stat_var, outcome.conversion_record),))
            if outcome.conversion_record is not None
            else None
        ),
        frame_grammar_version=FRAME_GRAMMAR_VERSION,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        mint_facets=mint.facets,
        family=request.family,
    )

    return pipeline.MaterialisationResult(
        frame_chunks=lambda: iter((outcome.frame,)),
        content_version=content_version,
        trace=trace,
        coverage=outcome.coverage,
        warnings=_build_warnings(request, outcome),
        series_statistics=series_statistics,
        output_profile=output_profile,
        conversion_record=outcome.conversion_record,
        member_conversion_records={request.stat_var: outcome.conversion_record},
        mint_preimage=MintPreimage(
            graph_revision=normalised_graph_revision,
            source_revisions=source_revisions,
            facets=mint.facets,
            conversion=ConversionPreimage.from_conversion_record(outcome.conversion_record),
        ),
        output_profile_seconds=output_profile_seconds,
        default_scope=outcome.default_scope.places,
        series_conversions=outcome.series_conversions,
        probe_empty=outcome.probe_empty,
    )


def _sorted_converted_members(
    member_conversion_records: Mapping[str, ConversionRecord | None],
) -> list[tuple[str, ConversionRecord]]:
    """Returns non-`None` conversion records sorted by StatVar. Converted members must share
    a recipe id and table-version pair or raise `MaterialisationUsageError`. Returns `[]` when
    no member converted."""
    converted = sorted(
        (
            (stat_var, record)
            for stat_var, record in member_conversion_records.items()
            if record is not None
        ),
        key=lambda pair: pair[0],
    )
    if not converted:
        return []
    recipe_ids = {record.recipe_id for _, record in converted}
    table_versions = {
        (record.exchange_rate_table_version, record.deflator_table_version)
        for _, record in converted
    }
    if len(recipe_ids) != 1 or len(table_versions) != 1:
        raise MaterialisationUsageError(
            "a group's converted members must share one conversion recipe and one pair "
            f"of table versions; found recipes {sorted(recipe_ids)!r} and table versions "
            f"{sorted(table_versions)!r}"
        )
    return converted


@dataclass(frozen=True)
class _GroupVersionMint:
    """A group's version and preimage inputs.

    `converted` holds the converted members sorted by StatVar and `representative` the
    first of their records, or `None` when no member converted."""

    content_version: str
    graph_revision: str
    source_revisions: tuple[dict[str, object], ...]
    facets: tuple[dict[str, object], ...]
    converted: list[tuple[str, ConversionRecord]]
    representative: ConversionRecord | None
    conversion: ConversionPreimage | None


def _mint_group_content_version(
    outcome: _GroupOutcome, *, revision_snapshot: RevisionSnapshot
) -> _GroupVersionMint:
    """Mints a group's version from `outcome.frame_sha256`, computed during the group's
    single fetch, avoiding another row read for hashing."""
    normalised = revision_snapshot.normalised()
    source_revisions = tuple(
        source_revision_preimage(source=sr.source, revision=sr.revision)
        for sr in normalised.source_revisions
    )
    facets = tuple(
        selection.as_grammar_facet()
        for selection in outcome.selections.values()
        if selection.ref is not None
    )

    converted = _sorted_converted_members(outcome.member_conversion_records)
    representative = converted[0][1] if converted else None
    representative_preimage = ConversionPreimage.from_conversion_record(representative)
    # One native-unit source revision per converted member, sorted by stat_var
    # (`_sorted_converted_members`' own order), mirroring `_finish_member`'s single
    # entry. Each member's native unit is `ConversionRecord.native_unit_ref`, the facet
    # unit `_finish_member` reads from `selections["unit"]`. `native_currency` is the
    # rate key and can name a twin (`CurrencyUSD` for a `USDollar` facet).
    source_revisions = source_revisions + tuple(
        native_unit_source_revision(
            stat_var=_bare_stat_var_ref(stat_var), native_unit=record.native_unit_ref
        )
        for stat_var, record in converted
    )
    if representative_preimage is not None:
        source_revisions += _recipe_body_revisions(representative_preimage)

    content_version = mint_version_from_digest(
        outcome.frame_sha256,
        graph_revision=normalised.graph_revision,
        source_revisions=source_revisions,
        facets=facets,
        conversion=representative_preimage,
    )
    return _GroupVersionMint(
        content_version=content_version,
        graph_revision=normalised.graph_revision,
        source_revisions=source_revisions,
        facets=facets,
        converted=converted,
        representative=representative,
        conversion=representative_preimage,
    )


def _finish_group(
    requests: Sequence[MaterialisationRequest],
    outcome: _GroupOutcome,
    *,
    revision_snapshot: RevisionSnapshot,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
) -> pipeline.MaterialisationResult:
    """Runs after the revision snapshot is known. Mints through `_mint_group_content_version`.
    The group's single fetch builds per-series statistics. The output profile consumes chunks
    one at a time.

    Sorts `resolved_stat_vars` by StatVar and looks up counts by StatVar, so zero-row members
    remain in the trace regardless of request order. The output profile uses the shared role
    columns, which the pipeline validates before this call. `output_profile_seconds` records
    profile computation time."""
    mint = _mint_group_content_version(outcome, revision_snapshot=revision_snapshot)
    content_version = mint.content_version
    source_revisions = mint.source_revisions
    facets = mint.facets
    converted = mint.converted
    representative_preimage = mint.conversion
    series_statistics = outcome.series_statistics
    profile_start = time.perf_counter()
    output_profile = compute_output_profile_from_chunks(
        outcome.frame_chunks,
        role_columns=requests[0].declared_roles,
        enrichment=enrichment,
    )
    output_profile_seconds = time.perf_counter() - profile_start

    resolved_stat_vars = tuple(
        ResolvedStatVar(
            ref=request.stat_var,
            roles=request.declared_roles,
            observation_count=outcome.member_observation_counts[request.stat_var],
        ).as_trace_entry()
        for request in sorted(requests, key=lambda r: r.stat_var)
    )

    trace = ManifestTrace(
        applied_window=(
            outcome.applied_window.as_trace_entry() if outcome.applied_window is not None else None
        ),
        graph_revision=mint.graph_revision,
        source_revisions=source_revisions,
        resolved_stat_vars=resolved_stat_vars,
        facets=tuple(selection.as_trace_entry() for selection in outcome.selections.values()),
        normalisations=(),
        conversion=_conversion_trace(converted),
        frame_grammar_version=FRAME_GRAMMAR_VERSION,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        mint_facets=facets,
        # Every request of a group is a member of one definition and carries its family.
        family=requests[0].family,
    )

    return pipeline.MaterialisationResult(
        frame_chunks=outcome.frame_chunks,
        content_version=content_version,
        trace=trace,
        coverage=outcome.coverage,
        warnings=_build_warnings(requests[0], outcome),
        series_statistics=series_statistics,
        output_profile=output_profile,
        conversion_record=None,
        member_conversion_records=outcome.member_conversion_records,
        mint_preimage=MintPreimage(
            graph_revision=mint.graph_revision,
            source_revisions=source_revisions,
            facets=facets,
            conversion=representative_preimage,
        ),
        output_profile_seconds=output_profile_seconds,
        default_scope=outcome.default_scope.places,
        series_conversions=outcome.series_conversions,
        probe_empty=outcome.probe_empty,
    )


def _build_warnings(
    request: MaterialisationRequest, outcome: _MemberOutcome | _GroupOutcome | _PreviewOutcome
) -> tuple[MaterialisationWarning, ...]:
    """Adds `empty_result` for zero rows, which suppresses `coverage_below_scope`. Its message
    says the request's places hold no observations when the facet probe returned no candidate
    (`_empty_result_warning`). `coverage_below_scope` covers window clamps and unserved bound entities, naming up to five entities, and a request
    binding no place that was served the countries with data in place of every place. One
    `coverage_below_scope` states every cause that applies.
    `deflator_projected` applies when any conversion record has `projected_count > 0`, and
    `currency_unmapped_omitted` when any record omitted an entity's rows; an omitted entity
    the request bound counts as reported there, so `coverage_below_scope` does not name it.
    `deflator_missing_omitted` applies when any record left out rows for a missing deflator.
    Warnings travel on the result. Member, group, and preview outcomes
    supply row counts and served-entity summaries without retaining survivor rows."""
    warnings: list[MaterialisationWarning] = []
    probe_empty = False
    if isinstance(outcome, _MemberOutcome):
        row_count = outcome.coverage.rows
        probe_empty = outcome.probe_empty
        converted_records = (
            (outcome.conversion_record,) if outcome.conversion_record is not None else ()
        )
    elif isinstance(outcome, _GroupOutcome):
        row_count = outcome.row_count
        probe_empty = outcome.probe_empty
        converted_records = tuple(
            record for record in outcome.member_conversion_records.values() if record is not None
        )
    else:
        row_count = outcome.total_rows
        converted_records = (
            (outcome.conversion_record,) if outcome.conversion_record is not None else ()
        )
    served = outcome.served_entities_by_role

    if row_count == 0:
        warnings.append(
            _empty_result_warning(request, group=isinstance(outcome, _GroupOutcome))
            if probe_empty
            else _warning("empty_result")
        )

    omitted = frozenset(
        entity for record in converted_records for entity in record.omitted_entities
    )
    unserved = [
        entity
        for entity in unserved_entities(_plain_role_bindings(request.role_bindings), served)
        if entity not in omitted
    ]
    countries_fallback = row_count > 0 and not outcome.default_scope.every_place
    if countries_fallback or coverage_below_scope(
        outcome.coverage, applied_window=outcome.applied_window, unserved=unserved
    ):
        warnings.append(
            _coverage_below_scope_warning(unserved, countries_fallback=countries_fallback)
        )

    if any(record.projected_count > 0 for record in converted_records):
        warnings.append(deflator_projected_warning(converted_records))

    if omitted:
        warnings.append(currency_unmapped_omitted_warning(omitted))

    deflated = [
        (entity, year)
        for record in converted_records
        for entity, years in record.deflator_omitted
        for year in years
    ]
    if deflated:
        warnings.append(deflator_missing_omitted_warning(deflated))

    return tuple(warnings)


# Warnings use a local `TypedDict` because no shared warning model exists.
# Refusal issues use the shared pydantic model.
def _warning(code: str) -> MaterialisationWarning:
    _title, detail = MATERIALISE_CODE_CATALOGUE[code]
    return {"code": code, "severity": "warn", "message": detail}


def _empty_result_warning(
    request: MaterialisationRequest, *, group: bool
) -> MaterialisationWarning:
    """`empty_result` for a result whose facet probe returned no candidate. It adds to the catalogue
    detail that the places the request covers hold no observations of its variable, or of a
    group's variables."""
    warning = _warning("empty_result")
    variables = "the definition's variables" if group else "the variable"
    where = (
        f"No place holds observations of {variables}."
        if binds_no_place(request.role_bindings, request.declared_roles)
        else f"The bound places hold no observations of {variables}."
    )
    return {**warning, "message": f"{warning['message']} {where}"}


# Maximum unserved dcids included in a warning.
_MAX_UNSERVED_NAMED = 5


def _coverage_below_scope_warning(
    unserved: Sequence[str], *, countries_fallback: bool
) -> MaterialisationWarning:
    """Adds up to five unserved dcids to the catalogue detail, then, with `countries_fallback`,
    that a request binding no place was served the countries with data because the full list
    of places with data was too large or unavailable (`place_scope.DefaultPlaceScope`). Without
    either the detail stands for a pure window clamp. Names appear as raw dcids."""
    _title, detail = MATERIALISE_CODE_CATALOGUE["coverage_below_scope"]
    message = detail
    if unserved:
        named = ", ".join(unserved[:_MAX_UNSERVED_NAMED])
        remaining = len(unserved) - _MAX_UNSERVED_NAMED
        if remaining > 0:
            named = f"{named} and {remaining} more"
        message = f"{message} No observations were served for {named}."
    if countries_fallback:
        message = (
            f"{message} No place was named, so the answer covers the countries with data, "
            "because the full list of places with data was too large or unavailable."
        )
    return {"code": "coverage_below_scope", "severity": "warn", "message": message}


class _ProjectedConversion(Protocol):
    """Requires projection counts and examples. Implementations may use different sequence types."""

    @property
    def projected_count(self) -> int: ...
    @property
    def projected_examples(self) -> Sequence[str]: ...


def deflator_projected_warning(
    records: Sequence[_ProjectedConversion],
) -> MaterialisationWarning:
    """Reports projected row count and sorted distinct `entity@year` examples across records.
    Examples are capped, with truncation based on distinct keys rather than row count."""
    _title, detail = MATERIALISE_CODE_CATALOGUE["deflator_projected"]
    total = sum(record.projected_count for record in records)
    examples, truncated = _capped_distinct_examples(
        example for record in records for example in record.projected_examples
    )
    truncated_note = " (truncated)" if truncated else ""
    message = f"{detail} {total} row(s), for example: {', '.join(examples)}{truncated_note}"
    return {"code": "deflator_projected", "severity": "warn", "message": message}


def currency_unmapped_omitted_warning(entities: Iterable[str]) -> MaterialisationWarning:
    """Reports how many distinct entities a conversion into each entity's own currency left
    out, naming them sorted and capped as `deflator_projected_warning` caps its examples."""
    _title, detail = MATERIALISE_CODE_CATALOGUE["currency_unmapped_omitted"]
    distinct = set(entities)
    named, truncated = _capped_distinct_examples(distinct)
    truncated_note = " (truncated)" if truncated else ""
    message = f"{detail} {len(distinct)} entities: {', '.join(named)}{truncated_note}"
    return {"code": "currency_unmapped_omitted", "severity": "warn", "message": message}


def deflator_missing_omitted_warning(omitted: Iterable[tuple[str, int]]) -> MaterialisationWarning:
    """Reports how many distinct entities a constant-price conversion left rows out of for a
    missing deflator, naming each as `entity` with its omitted years as spans
    (`entity (1990-2001, 2004)`), sorted and capped as `deflator_projected_warning` caps its
    examples."""
    _title, detail = MATERIALISE_CODE_CATALOGUE["deflator_missing_omitted"]
    by_entity: dict[str, set[int]] = {}
    for entity, year in omitted:
        by_entity.setdefault(entity, set()).add(year)
    named, truncated = _capped_distinct_examples(
        f"{entity} ({_year_spans(years)})" for entity, years in by_entity.items()
    )
    truncated_note = " (truncated)" if truncated else ""
    message = f"{detail} {len(by_entity)} entities: {', '.join(named)}{truncated_note}"
    return {"code": "deflator_missing_omitted", "severity": "warn", "message": message}


def _year_spans(years: Iterable[int]) -> str:
    """`years` as runs of consecutive years, `1990-2001, 2004`."""
    runs: list[list[int]] = []
    for year in sorted(set(years)):
        if runs and year == runs[-1][1] + 1:
            runs[-1][1] = year
        else:
            runs.append([year, year])
    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in runs)


def _plain_role_bindings(role_bindings: Mapping[str, EntityFilter]) -> dict[str, Sequence[str]]:
    """Returns plain DCID lists for coverage checks. `ContainedIn` scopes require graph
    expansion and stay outside this mapping."""
    return {role: filt for role, filt in role_bindings.items() if not isinstance(filt, ContainedIn)}
