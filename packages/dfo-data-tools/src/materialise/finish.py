"""Builds versions, warnings, statistics, profiles, and traces after generation returns.
`_finish_member` handles single members and `_finish_group` handles combined materialisations.
Previews reuse member warnings without a trace."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from dfo_frame_grammar import FRAME_GRAMMAR_VERSION, frame_sha256

from dc_sdmx import ContainedIn, EntityFilter

# `pipeline` imports this module during initialisation. `MaterialisationResult` resolves when
# `_finish_member` runs, after `pipeline` has loaded.
from materialise import pipeline
from materialise.codes import (
    MATERIALISE_CODE_CATALOGUE,
    CountedEntityYears,
    CountedRefs,
    MaterialisationWarning,
    MemberFields,
    YearRange,
)
from materialise.content_version import mint_version_from_digest
from materialise.convert import ConversionRecord
from materialise.dependencies import Dependencies
from materialise.enrichment import EnrichmentLookup
from materialise.errors import MaterialisationUsageError
from materialise.facets import FacetSelection, binds_no_place, selected_continued_series
from materialise.frame_build import fold_method_periods

# `_build_warnings` checks outcome classes at runtime. `generation` imports only the
# `MaterialisationRequest` annotation under `TYPE_CHECKING`, so these imports create no cycle.
from materialise.generation import (
    _GroupOutcome,
    _MemberOutcome,
    _PreviewOutcome,
    bare_stat_var_ref,
)
from materialise.profile import compute_output_profile, compute_output_profile_from_chunks
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from materialise.trace import ManifestTrace, ResolvedStatVar
from materialise.window import (
    AppliedWindow,
    RequestedWindow,
    coverage_below_scope,
    unserved_entities,
)

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


# A conversion's recipe id, rate-table version, and deflator-table version.
type _ConversionTables = tuple[str, str, str | None]


def _conversion_tables(record: ConversionRecord) -> _ConversionTables:
    return (record.recipe_id, record.exchange_rate_table_version, record.deflator_table_version)


def _conversion_trace_entry(
    tables: _ConversionTables, members: Sequence[tuple[str, ConversionRecord]]
) -> dict[str, object]:
    """Builds one `ManifestTrace.conversion` entry for the members converted under one recipe
    and table-version pair. `members` lists each distinct (bare StatVar ref,
    `ConversionRecord.native_unit_ref`) pair once, sorted, so records over the same StatVar
    and native unit name it once. Fallback and projected counts sum over every
    record, because each record counts its own definition's rows; their examples are
    deduplicated, sorted, and capped at `_MAX_TRACE_EXAMPLES`, with `truncated` reflecting
    distinct examples, not row counts. `deflator_table_version` is omitted when no deflator
    table joined. Fallback pairs are JSON lists."""
    recipe_id, rate_table_version, deflator_table_version = tables
    entry: dict[str, object] = {"recipe": recipe_id, "rate_table_version": rate_table_version}
    if deflator_table_version is not None:
        entry["deflator_table_version"] = deflator_table_version
    entry["members"] = [
        {"stat_var": stat_var, "native_unit": native_unit}
        for stat_var, native_unit in sorted(
            {(bare_stat_var_ref(stat_var), record.native_unit_ref) for stat_var, record in members}
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
    if left_out := _left_out(record for _, record in members):
        entry["left_out"] = left_out
    return entry


def _left_out(records: Iterable[ConversionRecord]) -> list[dict[str, object]]:
    """What the conversions in `records` left out, counted across them: the distinct entities
    no single currency names (`currency_unmapped`), and the distinct entities and entity-years
    a missing deflator left out (`deflator_missing`). A reason that left nothing out is
    absent."""
    unmapped: set[str] = set()
    deflated: set[tuple[str, int]] = set()
    for record in records:
        unmapped.update(record.omitted_entities)
        deflated.update(
            (entity, year) for entity, years in record.deflator_omitted for year in years
        )
    left_out: list[dict[str, object]] = []
    if unmapped:
        left_out.append({"reason": "currency_unmapped", "count": len(unmapped), "of": "entities"})
    if deflated:
        left_out.append(
            {
                "reason": "deflator_missing",
                "count": len({entity for entity, _ in deflated}),
                "of": "entities",
            }
        )
        left_out.append(
            {"reason": "deflator_missing", "count": len(deflated), "of": "entity_periods"}
        )
    return left_out


def _conversion_trace(
    converted: Iterable[tuple[str, ConversionRecord]],
) -> tuple[dict[str, object], ...] | None:
    """Builds `ManifestTrace.conversion`: one entry per distinct (recipe, rate-table version,
    deflator-table version), naming the members converted under it, or `None` when no member
    converted. Entries sort by recipe then table versions, and members by StatVar, so the
    trace is the same whatever order members arrive in. A group mixing recipes or table
    versions gets one entry for each."""
    by_tables: dict[_ConversionTables, list[tuple[str, ConversionRecord]]] = {}
    for stat_var, record in sorted(converted, key=lambda pair: pair[0]):
        by_tables.setdefault(_conversion_tables(record), []).append((stat_var, record))
    if not by_tables:
        return None
    return tuple(
        _conversion_trace_entry(tables, by_tables[tables])
        for tables in sorted(
            by_tables,
            key=lambda t: (t[0], t[1], t[2] is not None, t[2] or ""),
        )
    )


def _choices(selections: Iterable[FacetSelection]) -> tuple[dict[str, object], ...]:
    """`ManifestTrace.choices`: each selection's `choice_entry`, in axis order, for those the
    default policy chose over disagreeing candidates."""
    return tuple(entry for selection in selections if (entry := selection.choice_entry()))


def _continuations(
    selections: Iterable[FacetSelection], method_periods: Mapping[str, tuple[str, str]]
) -> tuple[dict[str, object], ...]:
    """`ManifestTrace.continuations`: for each selection whose series a method continues
    (`FacetSelection.continued_by`), its `axis`, the continuing `method`, and `periods`, the
    first and last period that method's rows supply in the served frame (`method_periods`),
    so after the reported value wins any period both hold. A continuing method that supplies
    no row is left out."""
    return tuple(
        {"axis": selection.axis, "method": method, "periods": list(method_periods[method])}
        for selection in selections
        if (method := selection.continued_by) is not None and method in method_periods
    )


def _source_revisions(dependencies: Dependencies) -> tuple[dict[str, object], ...]:
    """The trace's `source_revisions`: each dispatched provenance and rate-table provenance
    the run read, with the refresh token its dependency fingerprint records, sorted by source.
    A source whose refresh token the run could not read is left out."""
    tokens = dict(dependencies.sources) | dict(dependencies.conversion_sources)
    return tuple(
        {"source": source, "revision": token}
        for source, token in sorted(tokens.items())
        if token is not None
    )


def _finish_member(
    request: MaterialisationRequest,
    outcome: _MemberOutcome,
    *,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
    dependencies: Dependencies,
    graph_revision: str | None,
) -> pipeline.MaterialisationResult:
    """After generation returns, builds the version, warnings, per-series statistics, output
    profile, and trace. `output_profile_seconds` records the
    profile computation time."""
    frame_digest = frame_sha256(outcome.frame)
    content_version = mint_version_from_digest(frame_digest)
    # Statistics derive from the frame but do not enter its content-version preimage.
    statistics_sink = SeriesStatisticsSink(continuing=selected_continued_series(outcome.selections))
    statistics_sink.write(compute_series_statistics_table(outcome.frame))
    series_statistics = statistics_sink.finish()
    profile_start = time.perf_counter()
    output_profile = compute_output_profile(
        outcome.frame, role_columns=request.declared_roles, enrichment=enrichment
    )
    output_profile_seconds = time.perf_counter() - profile_start
    method_periods: dict[str, tuple[str, str]] = {}
    fold_method_periods(method_periods, outcome.frame)
    family = request.family_statement

    trace = ManifestTrace(
        applied_window=(
            outcome.applied_window.as_trace_entry() if outcome.applied_window is not None else None
        ),
        graph_revision=graph_revision,
        source_revisions=_source_revisions(dependencies),
        resolved_stat_vars=(
            ResolvedStatVar(
                ref=request.stat_var,
                roles=request.declared_roles,
                observation_count=outcome.coverage.rows,
            ).as_trace_entry(),
        ),
        facets=tuple(selection.as_trace_entry() for selection in outcome.selections.values()),
        normalisations=(),
        choices=_choices(outcome.selections.values()),
        continuations=_continuations(outcome.selections.values(), method_periods),
        conversion=(
            _conversion_trace(((request.stat_var, outcome.conversion_record),))
            if outcome.conversion_record is not None
            else None
        ),
        frame_grammar_version=FRAME_GRAMMAR_VERSION,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        family=family.members,
        stub_candidates=family.stub_candidates,
        coverage_gaps=tuple(gap.as_trace_entry() for gap in outcome.coverage_gaps),
    )

    return pipeline.MaterialisationResult(
        frame_chunks=lambda: iter((outcome.frame,)),
        frame_sha256=frame_digest,
        content_version=content_version,
        trace=trace,
        coverage=outcome.coverage,
        warnings=_build_warnings(request, outcome),
        series_statistics=series_statistics,
        output_profile=output_profile,
        conversion_record=outcome.conversion_record,
        member_conversion_records={request.stat_var: outcome.conversion_record},
        dependencies=dependencies,
        output_profile_seconds=output_profile_seconds,
        default_scope=outcome.default_scope.places,
        series_conversions=outcome.series_conversions,
        probe_empty=outcome.probe_empty,
        latest_years=outcome.latest_years,
        read_series=outcome.read_series,
        countries_fallback=_countries_fallback(outcome.coverage.rows, outcome),
    )


def _countries_fallback(row_count: int, outcome: _MemberOutcome | _GroupOutcome) -> bool:
    """Whether rows were served and a request binding no place was served the countries with
    data in place of every place (`place_scope.DefaultPlaceScope`)."""
    return row_count > 0 and not outcome.default_scope.every_place


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


def _finish_group(
    requests: Sequence[MaterialisationRequest],
    outcome: _GroupOutcome,
    *,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
    dependencies: Dependencies,
    graph_revision: str | None,
    absent_reads_zero: bool = False,
) -> pipeline.MaterialisationResult:
    """Runs after generation returns. Mints from `outcome.frame_sha256`, computed
    during the group's single fetch, so hashing rereads no rows. The group's single fetch builds
    per-series statistics. The output profile consumes chunks one at a time.

    Sorts `resolved_stat_vars` by StatVar and looks up counts by StatVar, so zero-row members
    remain in the trace regardless of request order. The output profile uses the shared role
    columns, which the pipeline validates before this call. `output_profile_seconds` records
    profile computation time."""
    content_version = mint_version_from_digest(outcome.frame_sha256)
    converted = _sorted_converted_members(outcome.member_conversion_records)
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

    # Every request of a group is a member of one definition, or one composite component,
    # and carries its family.
    family = requests[0].family_statement
    trace = ManifestTrace(
        applied_window=(
            outcome.applied_window.as_trace_entry() if outcome.applied_window is not None else None
        ),
        graph_revision=graph_revision,
        source_revisions=_source_revisions(dependencies),
        resolved_stat_vars=resolved_stat_vars,
        facets=tuple(selection.as_trace_entry() for selection in outcome.selections.values()),
        normalisations=(),
        choices=_choices(outcome.selections.values()),
        continuations=_continuations(outcome.selections.values(), outcome.method_periods),
        conversion=_conversion_trace(converted),
        frame_grammar_version=FRAME_GRAMMAR_VERSION,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        family=family.members,
        stub_candidates=family.stub_candidates,
    )

    return pipeline.MaterialisationResult(
        frame_chunks=outcome.frame_chunks,
        frame_sha256=outcome.frame_sha256,
        content_version=content_version,
        trace=trace,
        coverage=outcome.coverage,
        warnings=_build_warnings(requests[0], outcome, absent_reads_zero=absent_reads_zero),
        series_statistics=series_statistics,
        output_profile=output_profile,
        conversion_record=None,
        member_conversion_records=outcome.member_conversion_records,
        dependencies=dependencies,
        output_profile_seconds=output_profile_seconds,
        default_scope=outcome.default_scope.places,
        series_conversions=outcome.series_conversions,
        probe_empty=outcome.probe_empty,
        latest_years=outcome.latest_years,
        read_series=outcome.read_series,
        countries_fallback=_countries_fallback(outcome.row_count, outcome),
        unserved_members=outcome.unserved_members,
    )


def _build_warnings(
    request: MaterialisationRequest,
    outcome: _MemberOutcome | _GroupOutcome | _PreviewOutcome,
    *,
    absent_reads_zero: bool = False,
) -> tuple[MaterialisationWarning, ...]:
    """Adds `empty_result` for zero rows, which suppresses `coverage_below_scope`. Its message
    says the request's places hold no observations when the facet probe returned no candidate
    (`_empty_result_warning`). `coverage_below_scope` covers window clamps, unserved bound
    entities, a request binding no place that was served the countries with data in place of
    every place, and, when the group served rows, its members that served nothing. It counts
    the entities and members in its message and names them by ref in its structured fields.
    The members are those at another observation period in the requested years
    (`_GroupOutcome.off_cadence_members`), and those holding observations only outside them
    (`_GroupOutcome.out_of_window_members`), and those holding no facet of the elected group
    (`_GroupOutcome.unserved_members`). The second are left out with `absent_reads_zero`,
    when the served dataset reads an absent member as zero, since such a member adds nothing
    to the requested years. One `coverage_below_scope` states every cause that applies.
    `currency_unmapped_omitted` applies when any conversion record omitted an entity's rows;
    an omitted entity the request bound counts as reported there, so `coverage_below_scope`
    does not count it. `deflator_missing_omitted` applies when any record left out rows for a
    missing deflator. Values priced from a projected rate or deflator raise no warning: the
    trace's `conversion[].projected` counts them, and a description states them as an
    `estimated` caveat.
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
    countries_fallback = (
        row_count > 0 and not outcome.default_scope.every_place
        if isinstance(outcome, _PreviewOutcome)
        else _countries_fallback(row_count, outcome)
    )
    off_cadence: Sequence[str] = ()
    out_of_window: Sequence[str] = ()
    unserved_members: Sequence[str] = ()
    member_count = 0
    if isinstance(outcome, _GroupOutcome) and row_count:
        off_cadence = outcome.off_cadence_members
        out_of_window = () if absent_reads_zero else outcome.out_of_window_members
        unserved_members = outcome.unserved_members
        member_count = len(outcome.member_observation_counts)
    if writes_coverage_below_scope(
        below_scope=coverage_below_scope(
            outcome.coverage, applied_window=outcome.applied_window, unserved=unserved
        ),
        countries_fallback=countries_fallback,
        off_cadence=off_cadence,
        out_of_window=out_of_window,
        unserved_members=unserved_members,
    ):
        warnings.append(
            _coverage_below_scope_warning(
                unserved,
                countries_fallback=countries_fallback,
                off_cadence=off_cadence,
                out_of_window=out_of_window,
                unserved_members=unserved_members,
                member_count=member_count,
                years=_clamped_years(request.window, outcome.applied_window),
            )
        )

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


def writes_coverage_below_scope(
    *,
    below_scope: bool,
    countries_fallback: bool,
    off_cadence: Sequence[str],
    out_of_window: Sequence[str],
    unserved_members: Sequence[str],
) -> bool:
    """Whether a mint writes `coverage_below_scope`: the window was clamped or a bound entity
    went unserved (`below_scope`, `window.coverage_below_scope`), a request binding no place
    was served the countries with data, or a group left members out. `_build_warnings` and
    the stored-warning backfill (`datasets_api.warning_backfill`) both ask this."""
    return (
        below_scope or countries_fallback or bool(off_cadence or out_of_window or unserved_members)
    )


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


# Maximum refs a warning's counted field carries. A conversion's left-out entities keep the
# ten the warnings named in their message, `_MAX_TRACE_EXAMPLES`.
MAX_COUNTED_REFS = 5

# Entities are bare dcids; a warning's counted field carries them as `dcid:` refs, as it
# carries a group's members.
_DCID_REF_PREFIX = "dcid:"


# The sentence of `coverage_below_scope` saying a request binding no place was served the
# countries with data.
COUNTRIES_FALLBACK_SENTENCE = (
    "No place was named, so the answer covers the countries with data, because the full list "
    "of places with data was too large or unavailable."
)


def _clamped_years(
    requested: RequestedWindow, applied: AppliedWindow | None
) -> tuple[YearRange, YearRange] | None:
    """The years a range asked for and the years its data covers, when the data covers fewer
    (`AppliedWindow.origin` is `coverage_clamp`). An end the definition left unstated
    (`RequestedWindow.end_unstated`) reads as open, and an open bound is left out. None for
    any other window."""
    if applied is None or applied.origin != "coverage_clamp":
        return None
    asked: YearRange = {}
    if requested.start_year is not None:
        asked["start"] = requested.start_year
    if requested.end_year is not None and not requested.end_unstated:
        asked["end"] = requested.end_year
    served: YearRange = {}
    if applied.start_year is not None:
        served["start"] = applied.start_year
    if applied.end_year is not None:
        served["end"] = applied.end_year
    return asked, served


def _year_range_text(years: YearRange) -> str:
    """`2020 to 2030`, `2020 onward` for an open end, or `up to 2030` for an open start."""
    start, end = years.get("start"), years.get("end")
    if start is None:
        return f"up to {end}"
    if end is None:
        return f"{start} onward"
    return f"{start} to {end}"


def _coverage_below_scope_warning(
    unserved: Sequence[str],
    *,
    countries_fallback: bool,
    off_cadence: Sequence[str],
    out_of_window: Sequence[str],
    unserved_members: Sequence[str],
    member_count: int,
    years: tuple[YearRange, YearRange] | None = None,
) -> MaterialisationWarning:
    """With `years`, the requested range and the years its data covers (`_clamped_years`),
    it states both after the catalogue detail, as "Requested 2020 to 2030; the data covers
    2020 to 2024.", and carries them as `requested_years` and `served_years`. Then it counts
    the `unserved` entity dcids. Then, of the group's `member_count` series, it counts those in `off_cadence`, which report the requested years
    at another frequency than the group serves and are left out, and those in
    `out_of_window`, which hold no observation in the requested years, and those in
    `unserved_members`, which hold no observation under the source, unit and method the
    group serves and are left out. The message names no entity or member, so a stored
    version holds no name that can go stale. The warning's `unserved_entities`,
    `off_cadence_members`, `out_of_window_members` and `unserved_members` carry each count
    and the first five refs (`CountedRefs`), beside `member_count` for the members, for a
    reader to name them. Then, with `countries_fallback`, it says a request binding no place
    was served the countries with data because the full list of places with data was too
    large or unavailable (`place_scope.DefaultPlaceScope`). Without any the detail stands
    alone."""
    _title, detail = MATERIALISE_CODE_CATALOGUE["coverage_below_scope"]
    message = detail
    if years is not None:
        requested, served = years
        message = (
            f"{message} Requested {_year_range_text(requested)}; "
            f"the data covers {_year_range_text(served)}."
        )
    if unserved:
        message = (
            f"{message} No observations were served for {len(unserved)} requested "
            f"{_entities(len(unserved))}."
        )
    for sentence in member_sentences(
        off_cadence=off_cadence,
        out_of_window=out_of_window,
        unserved_members=unserved_members,
        member_count=member_count,
    ):
        message = f"{message} {sentence}"
    if countries_fallback:
        message = f"{message} {COUNTRIES_FALLBACK_SENTENCE}"
    warning: MaterialisationWarning = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": message,
    }
    if years is not None:
        warning["requested_years"], warning["served_years"] = years
    if unserved:
        warning["unserved_entities"] = _counted_entities(unserved)
    warning.update(
        member_fields(
            off_cadence=off_cadence,
            out_of_window=out_of_window,
            unserved_members=unserved_members,
            member_count=member_count,
        )
    )
    return warning


def member_sentences(
    *,
    off_cadence: Sequence[str],
    out_of_window: Sequence[str],
    unserved_members: Sequence[str],
    member_count: int,
) -> tuple[str, ...]:
    """The sentences of a group's `coverage_below_scope` that count, of its `member_count`
    series, those in `off_cadence`, `out_of_window` and `unserved_members`, one sentence for
    each that is not empty (`_coverage_below_scope_warning`)."""
    sentences: list[str] = []
    if off_cadence:
        sentences.append(
            f"{len(off_cadence)} of {member_count} series report at a different "
            "frequency and are left out of the figures."
        )
    if out_of_window:
        sentences.append(
            f"{len(out_of_window)} of {member_count} series hold no observations "
            "in the requested years."
        )
    if unserved_members:
        sentences.append(
            f"{len(unserved_members)} of {member_count} series hold no observations "
            "under the source, unit and method the figures use and are left out."
        )
    return tuple(sentences)


def member_fields(
    *,
    off_cadence: Sequence[str],
    out_of_window: Sequence[str],
    unserved_members: Sequence[str],
    member_count: int,
) -> MemberFields:
    """The fields of a group's `coverage_below_scope` naming the members `member_sentences`
    counts: `member_count` when any is named, and each non-empty kind's count and first five
    refs (`CountedRefs`)."""
    fields: MemberFields = {}
    if off_cadence or out_of_window or unserved_members:
        fields["member_count"] = member_count
    if off_cadence:
        fields["off_cadence_members"] = _counted(off_cadence)
    if out_of_window:
        fields["out_of_window_members"] = _counted(out_of_window)
    if unserved_members:
        fields["unserved_members"] = _counted(unserved_members)
    return fields


def _counted(refs: Sequence[str], *, limit: int = MAX_COUNTED_REFS) -> CountedRefs:
    return {"count": len(refs), "refs": list(refs[:limit])}


def _counted_entities(entities: Iterable[str], *, limit: int = MAX_COUNTED_REFS) -> CountedRefs:
    """The distinct bare `entities`, sorted, as `dcid:` refs, the first `limit` of them
    carried (`_counted`)."""
    return _counted(
        [f"{_DCID_REF_PREFIX}{entity}" for entity in sorted(set(entities))], limit=limit
    )


def _counted_entity_years(omitted: Iterable[tuple[str, int]]) -> CountedEntityYears:
    """The distinct bare entities of `omitted`, counted, and the first `_MAX_TRACE_EXAMPLES`,
    sorted, as `dcid:` refs each with its years (`_year_spans`)."""
    by_entity: dict[str, set[int]] = {}
    for entity, year in omitted:
        by_entity.setdefault(entity, set()).add(year)
    return {
        "count": len(by_entity),
        "entities": [
            {"ref": f"{_DCID_REF_PREFIX}{entity}", "years": _year_spans(by_entity[entity])}
            for entity in sorted(by_entity)[:_MAX_TRACE_EXAMPLES]
        ],
    }


def _year_spans(years: Iterable[int]) -> list[str]:
    """`years` as runs of consecutive years, `["1990-2001", "2004"]`."""
    runs: list[list[int]] = []
    for year in sorted(set(years)):
        if runs and year == runs[-1][1] + 1:
            runs[-1][1] = year
        else:
            runs.append([year, year])
    return [str(first) if first == last else f"{first}-{last}" for first, last in runs]


def _entities(count: int) -> str:
    return "entity" if count == 1 else "entities"


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
    Examples are capped, with truncation based on distinct keys rather than row count. Search's
    observation answers raise it; a materialisation's warnings do not."""
    _title, detail = MATERIALISE_CODE_CATALOGUE["deflator_projected"]
    total = sum(record.projected_count for record in records)
    examples, truncated = _capped_distinct_examples(
        example for record in records for example in record.projected_examples
    )
    truncated_note = " (truncated)" if truncated else ""
    message = f"{detail} {total} row(s), for example: {', '.join(examples)}{truncated_note}"
    return {"code": "deflator_projected", "severity": "warn", "message": message}


def currency_unmapped_omitted_warning(entities: Iterable[str]) -> MaterialisationWarning:
    """Counts the distinct entities a conversion into each entity's own currency left out.
    `omitted_entities` carries the first `_MAX_TRACE_EXAMPLES`, sorted (`_counted_entities`)."""
    _title, detail = MATERIALISE_CODE_CATALOGUE["currency_unmapped_omitted"]
    counted = _counted_entities(entities, limit=_MAX_TRACE_EXAMPLES)
    return {
        "code": "currency_unmapped_omitted",
        "severity": "warn",
        "message": f"{detail} It left out {counted['count']} {_entities(counted['count'])}.",
        "omitted_entities": counted,
    }


def deflator_missing_omitted_warning(omitted: Iterable[tuple[str, int]]) -> MaterialisationWarning:
    """Counts the distinct entities a constant-price conversion left rows out of for a
    missing deflator. `omitted_entity_years` carries the first `_MAX_TRACE_EXAMPLES`, sorted,
    each with the years of those rows (`_counted_entity_years`)."""
    _title, detail = MATERIALISE_CODE_CATALOGUE["deflator_missing_omitted"]
    counted = _counted_entity_years(omitted)
    return {
        "code": "deflator_missing_omitted",
        "severity": "warn",
        "message": (
            f"{detail} It left out rows of {counted['count']} {_entities(counted['count'])}."
        ),
        "omitted_entity_years": counted,
    }


def _plain_role_bindings(role_bindings: Mapping[str, EntityFilter]) -> dict[str, Sequence[str]]:
    """Returns plain DCID lists for coverage checks. `ContainedIn` scopes require graph
    expansion and stay outside this mapping."""
    return {role: filt for role, filt in role_bindings.items() if not isinstance(filt, ContainedIn)}
