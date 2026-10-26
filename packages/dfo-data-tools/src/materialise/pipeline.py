"""Materialisation entry points, result types and request validation.

A mint generates its data, then finishes it into a frame, trace and content version. A preview can
mint the same content version over its full survivor frame.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from code_index.types import ComponentRole
from dc_sdmx import EntityFilter, SdmxClient
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.codes import MaterialisationWarning
from materialise.content_version import mint_version
from materialise.convert import ConversionRecord
from materialise.dependencies import Dependencies, build_dependencies
from materialise.enrichment import EnrichmentLookup
from materialise.errors import FacetBindingUnsupportedError, MaterialisationUsageError
from materialise.facets import (
    CollectionFacetProbe,
    ElectedDataset,
    FacetBinding,
    FacetSelection,
    VariableBreadth,
    admitted_candidates,
    probe_facets,
)
from materialise.finish import (
    _build_warnings,
    _finish_group,
    _finish_member,
)
from materialise.frame_build import Frame
from materialise.generation import (
    _finish_preview_member,
    _flatten_bound_entities,
    _generate_group,
    _generate_member,
    _prepare_preview_member,
    _PreparedPreview,
    _PreviewOutcome,
    _select_member_facets,
    bare_stat_var_ref,
    dispatched_provenances,
)
from materialise.graph import Facet, GraphReader
from materialise.member_conversion import decide_member_conversion, refuse_if_every_row_omitted
from materialise.profile import OutputProfile
from materialise.progress import RowProgress
from materialise.rate_tables import RateTableCache
from materialise.reshape import TableLike
from materialise.series_conversion import SeriesConversion, SeriesKey
from materialise.statistics import SeriesStatisticsFile
from materialise.trace import ManifestTrace, MemberFamily
from materialise.window import Coverage, LatestYear, ReadSeries, RequestedWindow
from origin_catalogue import note_un_unserved

if TYPE_CHECKING:
    # Keep the optional coverage index lazy because it imports NumPy, absent from pure installs.
    from series_coverage.index import CoverageIndex

# Deployment code supplies its configured bound; this is the package fallback.
DEFAULT_MEMBER_CONCURRENCY = 4


@dataclass(frozen=True)
class MaterialisationRequest:
    """Materialisation input after higher-level components adapt a canonical Spec. One facet probe
    covers all bound entities for each StatVar.
    """

    # Bare StatVar dcids raise `MaterialisationUsageError`; callers supply the `dcid:` prefix.
    stat_var: str
    role_bindings: Mapping[str, EntityFilter]
    # Declared roles remain available when `role_bindings` is empty (which means all entities).
    declared_roles: tuple[str, ...]
    # Keys are a subset of `facets.FACET_AXES`. A key absent from this
    # mapping is treated as `FacetBinding(state="unbound")`. All three axes
    # in `facets.FACET_AXES` are always evaluated, regardless of which keys
    # this mapping carries. This is the one place an axis and its binding
    # are known together, so `__post_init__` is where a `state="request"` or
    # `state="national_currency"` binding stored under any key other than
    # `"unit"` is caught -- a `FacetBinding` itself carries no axis of its own.
    facet_bindings: Mapping[str, FacetBinding]
    window: RequestedWindow
    # Code-index-derived shapes use `(provenance, unit)` for group facet agreement.
    index_sourced: bool = False
    # The dataset elected for an index-resolved family. Facet selection reads its
    # provenance alone (`facets.dataset_candidates`).
    dataset: ElectedDataset | None = None
    # The whole population the definition's identity admits, shared by every request of the
    # definition and stated in the trace. `None` when the resolver could not state it.
    family: MemberFamily | None = None
    # The `dcid:` refs member resolution left out of that family as stubs, nodes whose complete
    # read carries no definition arc, shared by every request of the definition and stated in
    # the trace.
    stub_candidates: tuple[str, ...] = ()
    # A caller's facet filter over each probed candidate's provenance, method and unit. The
    # member elects only among the candidates it admits (`facets.admitted_candidates`), so
    # a filtered request serves its best admitted facet. None admits every candidate.
    candidate_filter: Callable[[Facet], bool] | None = None
    # A read whose product must cite a source. With it, an unbound provenance axis refuses a
    # winning facet that carries none (`DefaultFacetUndefinedError`); without it the axis
    # resolves to the null ref as `measurementMethod` and `unit` do, and the answer states
    # no source.
    requires_provenance: bool = True
    # The component of a composite definition the request reads
    # (`code_index.composites.component_identities`), or None for a plain definition. The
    # requests of one component materialise together, and a composite's components compose
    # into one frame (`materialise.composite.compose_components`).
    component: ComponentRole | None = None

    def __post_init__(self) -> None:
        for axis, binding in self.facet_bindings.items():
            if binding.state in ("request", "national_currency") and axis != "unit":
                raise MaterialisationUsageError(
                    f"a {binding.state}-state facet binding only applies to the unit axis, "
                    f"not {axis!r}"
                )


@dataclass(frozen=True)
class MaterialisationResult:
    """`frame_chunks` returns a fresh iterator of validated frames in increasing `row_sort_key`
    order. Group chunks read from Parquet, so a caller needing two passes calls it twice and peak
    memory stays at one chunk. `frame_sha256` is the hash of those frames
    (`dfo_frame_grammar.frame_sha256`), and `content_version` is minted from it alone.
    `conversion_record` is set for a single member and `None` for a group or an unconverted result.
    `member_conversion_records` maps each StatVar to its record or `None`.
    `series_conversions` holds one record per converted series, keyed by bare StatVar and
    role tuple (`materialise.series_conversion`), and is empty for an unconverted result.
    `default_scope` is the places the default place scope resolved when a request binds no
    entity (`materialise.place_scope`), and empty otherwise.
    `probe_empty` is true when no member's facet probe returned a candidate, so the places the
    requests cover hold no observation of their variables. The result then holds zero rows, an
    `empty_result` warning and no facet selection.
    `dependencies` is the run's dependency fingerprint (`materialise.dependencies`), built from
    what it dispatched, so identical inputs give identical fingerprints whoever ran them. Its
    source tokens are the refresh dates the generation read before its first SDMX request.
    `latest_years` maps each request's `stat_var` to the year a `latest` window served it
    (`window.LatestYear`): the year the coverage index elected for it, the facets and
    omissions that election counted, the year its rows hold, and whether it fell back to
    the newest period any series holds. Empty for every other window.
    `read_series` maps each request's `stat_var` to the series its read holds under any
    window (`window.ReadSeries`): the facets it dispatched and the rows a conversion left
    out. No dependency fingerprint holds it outside a `latest` window's `latest_years`.
    `countries_fallback` is true when a request binding no place was served the countries
    with data in place of every place (`place_scope.DefaultPlaceScope`) and rows were served.
    """

    frame_chunks: Callable[[], Iterator[Frame]]
    frame_sha256: str
    content_version: str
    trace: ManifestTrace
    coverage: Coverage
    warnings: tuple[MaterialisationWarning, ...]
    series_statistics: SeriesStatisticsFile
    output_profile: OutputProfile
    conversion_record: ConversionRecord | None
    member_conversion_records: Mapping[str, ConversionRecord | None]
    dependencies: Dependencies
    # Time spent computing the output profile, in seconds.
    output_profile_seconds: float = 0.0
    default_scope: tuple[str, ...] = ()
    series_conversions: Mapping[SeriesKey, SeriesConversion] = field(default_factory=dict)
    probe_empty: bool = False
    latest_years: Mapping[str, LatestYear] = field(default_factory=dict)
    read_series: Mapping[str, ReadSeries] = field(default_factory=dict)
    countries_fallback: bool = False


@dataclass(frozen=True)
class ConcatenatedFrameChunks:
    """In-memory `TableLike` view that preserves row order. `render_and_store` uses it for no-op
    groupby without a pivot. Streaming groupby and pivot fold over chunks.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]

    @classmethod
    def from_chunks(cls, chunks: Iterator[TableLike]) -> ConcatenatedFrameChunks:
        rows: list[tuple[object, ...]] = []
        columns: tuple[str, ...] = ()
        for chunk in chunks:
            if not columns:
                columns = chunk.columns
            rows.extend(chunk.rows)
        return cls(columns=columns, rows=tuple(rows))


@dataclass(frozen=True)
class PreviewResult:
    """Write-free preview of one request's observations, with no trace.

    With `with_content_version`, `content_version` matches a mint for the same request. Without
    it, the version is `None`. The version covers the full survivor set regardless of `limit`.

    `total_rows` and `coverage` describe all surviving rows. `frame` keeps at most the `limit`
    most recent rows, in the frame grammar's order (by role dcids, then period). `truncated` is
    true when rows were dropped. `selections` records each axis's chosen reference, dispatch
    facet and basis.

    A read with `series_roles` cuts each series on its own: `frame` keeps each series' most
    recent `series_limit` rows, so it holds up to `series_limit` rows for every series the read
    returned, `series_total_rows` holds each series' surviving rows before the cut, keyed by
    the series' dcids on those roles, and `truncated` is true when any series was cut.

    `conversion_record` identifies arithmetic performed. `conversion_recipe_id` records the
    resolved recipe even when no arithmetic ran. Both are `None` for ref or unbound unit bindings.
    `default_scope` is `MaterialisationResult.default_scope`.
    `series_conversion_records` holds each series' record for a preview read with
    `series_roles`, keyed by the series' dcids on those roles.
    """

    frame: Frame
    coverage: Coverage
    total_rows: int
    truncated: bool
    conversion_record: ConversionRecord | None
    conversion_recipe_id: str | None
    selections: Mapping[str, FacetSelection]
    warnings: tuple[MaterialisationWarning, ...]
    content_version: str | None = None
    default_scope: tuple[str, ...] = ()
    series_conversion_records: Mapping[tuple[str, ...], ConversionRecord] = field(
        default_factory=dict
    )
    series_total_rows: Mapping[tuple[str, ...], int] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedPreview:
    """An opaque preview ready for optional exact-period finalisation."""

    _prepared: _PreparedPreview

    @property
    def latest_period(self) -> str | None:
        return self._prepared.latest_period

    @property
    def elected_period(self) -> str | None:
        """The newest period the read holds in the year it was told its StatVar elected,
        None when it was told none or holds no row in that year."""
        return self._prepared.elected_period

    @property
    def selections(self) -> Mapping[str, FacetSelection]:
        """Selected facets, for a caller validating its planned dispatch group."""
        return self._prepared.selections


def materialise_group(
    requests: Sequence[MaterialisationRequest],
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
    member_concurrency: int = DEFAULT_MEMBER_CONCURRENCY,
    rate_cache: RateTableCache | None = None,
    precomputed_probe: CollectionFacetProbe | None = None,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
    progress: RowProgress | None = None,
    absent_reads_zero: Callable[[str], bool] | None = None,
) -> MaterialisationResult:
    """Generate and finish a group. Returns one frame, trace and content version covering every
    member.

    Before any network call, raises `MaterialisationUsageError` for an empty request, duplicate
    StatVars, different declared-role sets or conflicting unit bindings, and
    `FacetBindingUnsupportedError` for a unit bound to each place's national currency across
    several members. Other facet-axis mismatches are graph-state refusals when a member lacks
    the requested facet.

    A unit bound to each place's national currency serves each place the facets in its own
    currency, unconverted (`national_currency.narrow_to_national_currency`), and the trace
    names the places it serves no row at (`ManifestTrace.coverage_gaps`).

    When no member's facet probe returns a candidate, the result is empty and says so
    (`MaterialisationResult.probe_empty`): no facet is selected and no conversion is planned.
    A group with a candidate on some member selects among those members' facets as usual.

    One request uses the member path. Multiple requests select a shared facet, group members with
    the same entity scope and `latest` status, and fetch each chunk with one SDMX call. At most
    `member_concurrency` chunks run at once. The aggregate row ceiling applies to the group.

    A `latest` window serves each role-bearing member at the year `coverage` elects for it
    (`CoverageIndex.elected_years`) over the facets the member dispatches, under its bound
    roles and the request's source scope, less the entities a conversion omits. The read
    stays `LATEST`, and a member with a series ending after its year is read once more at
    that year. A member whose elected year no row holds serves the newest period any of its
    series holds, as a member that elects no year does, and `latest_years` records both.

    `precomputed_probe` must match requests in order, and generation reuses it. What its UN reads
    could not serve joins the caller's `recording_un_blocked_places` record. Groups above the
    probe-pair limit require a supplied probe. `deadline` is one absolute monotonic bound shared by
    every SDMX stream and retry.

    Request-form unit bindings raise `MaterialisationUsageError` before fetch when `rate_cache` is
    absent. Role-bearing requests require `coverage` for facet counts and SDMX dispatch. Missing
    coverage raises the same error. `progress` increments
    by rows entering frames. The caller supplies deployment-level `materialiser_build` and
    `capability_version`.

    Every member converts against one version of each rate table, because the call loads tables
    through a `RateTableCache.for_request` view. A conversion into each entity's own currency
    omits the rows of entities no single currency names, and a constant-price conversion the
    rows whose deflator the table lacks; when no row of the group survives and some were
    omitted, the call refuses `currency_unmapped` naming the unnamed entities, else
    `deflator_missing` naming each entity and year (`refuse_if_every_row_omitted`).

    `absent_reads_zero` says whether a provenance reads an absent member as zero
    (`code_index.members.absent_member_reads_zero`), the rule a sum over the frame reads. A
    group served from one provenance it holds true for raises no warning for its members
    holding no observation in the requested years (`finish._build_warnings`). Without it
    every provenance reads unknown.
    """
    if not requests:
        raise MaterialisationUsageError("materialise_group requires at least one member")
    _check_group_role_agreement(requests)
    _check_group_stat_var_uniqueness(requests)
    _check_group_unit_binding_agreement(requests)
    _refuse_group_national_currency(requests)
    rate_cache = rate_cache.for_request() if rate_cache is not None else None
    if precomputed_probe is not None:
        note_un_unserved(precomputed_probe.un_unserved)
    # The graph serving revision the run reads, which its trace records.
    graph_revision = graph.generation_stamp().token

    if len(requests) == 1:
        outcome = _generate_member(
            requests[0],
            sdmx=sdmx,
            graph=graph,
            ceiling=ceiling,
            rate_cache=rate_cache,
            coverage=coverage,
            precomputed_probe=precomputed_probe,
            deadline=deadline,
            progress=progress,
        )
        refuse_if_every_row_omitted(
            served_rows=outcome.coverage.rows, records=(outcome.conversion_record,)
        )
        return _finish_member(
            requests[0],
            outcome,
            graph_revision=graph_revision,
            materialiser_build=materialiser_build,
            capability_version=capability_version,
            enrichment=enrichment,
            dependencies=build_dependencies(
                requests,
                source_refs=dispatched_provenances(outcome.selections),
                provenance_refresh=outcome.provenance_refresh,
                conversion_records=(outcome.conversion_record,),
                latest_years=outcome.latest_years,
            ),
        )

    outcome = _generate_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        ceiling=ceiling,
        member_concurrency=member_concurrency,
        rate_cache=rate_cache,
        coverage_index=coverage,
        precomputed_probe=precomputed_probe,
        deadline=deadline,
        progress=progress,
    )
    refuse_if_every_row_omitted(
        served_rows=outcome.coverage.rows, records=outcome.member_conversion_records.values()
    )
    served = dispatched_provenances(outcome.selections)
    return _finish_group(
        requests,
        outcome,
        graph_revision=graph_revision,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        enrichment=enrichment,
        absent_reads_zero=(
            absent_reads_zero is not None and len(served) == 1 and absent_reads_zero(served[0])
        ),
        dependencies=build_dependencies(
            requests,
            source_refs=served,
            provenance_refresh=outcome.provenance_refresh,
            conversion_records=outcome.member_conversion_records.values(),
            latest_years=outcome.latest_years,
        ),
    )


def materialise_spec(
    request: MaterialisationRequest,
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    materialiser_build: str,
    capability_version: str,
    enrichment: EnrichmentLookup,
    rate_cache: RateTableCache | None = None,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
    progress: RowProgress | None = None,
) -> MaterialisationResult:
    """Single-member form of `materialise_group`. Empty filtered results return a result with the
    `empty_result` warning and no applied window.

    Only request-form unit bindings use `rate_cache`.
    `coverage` and `progress` pass through to `materialise_group`. The caller supplies the deployed
    `materialiser_build` and `capability_version`.
    """
    return materialise_group(
        [request],
        sdmx=sdmx,
        graph=graph,
        ceiling=ceiling,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        enrichment=enrichment,
        rate_cache=rate_cache,
        coverage=coverage,
        deadline=deadline,
        progress=progress,
    )


def decide_member_unit(
    request: MaterialisationRequest,
    *,
    graph: GraphReader,
    sdmx: SdmxClient,
    coverage: CoverageIndex,
    settle: bool,
    precomputed: Sequence[Facet] | None = None,
    breadth: Mapping[str, int] | None = None,
    variable_breadth: VariableBreadth = None,
    deadline: float | None = None,
) -> Mapping[str, FacetSelection]:
    """Return the facets and conversion decision a mint would use. The function reads facet
    candidates but stops before observations and rate-table loading. `precomputed` skips the
    probe and ranks with `breadth`, each facet's series count under the request's window, and
    `variable_breadth`, each facet's places across the whole variable, as the probe's own
    counts rank a probed decision (`facets.MemberProbe.breadth_in`,
    `facets.MemberProbe.variable_breadth`). `settle`
    goes to the probe (`facets.probe_facets`), and Search passes `False`.

    Role-bearing requests require `coverage` or raise `MaterialisationUsageError` before probing.
    Selection can raise `DefaultFacetUndefinedError` or `FacetNotAvailableError`. Request-form
    units can raise `RateCoverageGapError`, `NonCurrencyUnitError` or `RecipeNotDeclaredError`.
    Ref and unbound bindings return selected facets only.
    """
    if precomputed is None:
        probe = probe_facets(
            stat_var=bare_stat_var_ref(request.stat_var),
            bound_entities=_flatten_bound_entities(request.role_bindings),
            graph=graph,
            availability=sdmx,
            settle=settle,
            declared_roles=request.declared_roles,
            role_bindings=request.role_bindings,
            coverage=coverage,
            window=request.window,
            deadline=deadline,
        )
        probed, breadth = probe.candidates, probe.breadth_in(request.window)
        variable_breadth = probe.variable_breadth
    else:
        # A decision fetches no rows, so precomputed candidates need no admission.
        probed = precomputed
    candidates = admitted_candidates(probed, request.candidate_filter)
    selections = _select_member_facets(
        request, candidates, graph=graph, breadth=breadth, variable_breadth=variable_breadth
    ).selections
    decide_member_conversion(request=request, selections=selections, graph=graph, deadline=deadline)
    return selections


def prepare_materialise_preview(
    request: MaterialisationRequest,
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    limit: int,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    rate_cache: RateTableCache | None = None,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
    precomputed: Sequence[Facet] | None = None,
    series_roles: Sequence[str] = (),
    series_limit: int | None = None,
    allowed_series_keys: frozenset[tuple[str, ...]] | None = None,
    elected_year: int | None = None,
) -> PreparedPreview:
    """Read and reduce a preview without applying conversion arithmetic.

    ``finish_materialise_preview`` may select one exact period before it
    converts.  This is intended for coordinated Search latest reads.

    `elected_year` is the year the caller's plan elected for the StatVar under a `latest`
    window, None when it elected none. The read is a mint's (`materialise_group`): `LATEST`
    rows, and the rows at that year when a series ends after it. The prepared rows hold
    that year's period beside the newest period of each series, so the caller can serve
    the year when any read of the StatVar holds it (`PreparedPreview.elected_period`) and
    the newest period otherwise. The read itself elects nothing from `coverage`.
    """
    if limit < 1:
        raise MaterialisationUsageError(f"materialise_preview requires limit >= 1, got {limit!r}")
    if series_roles and (series_limit is None or series_limit < 1):
        raise MaterialisationUsageError(
            "materialise_preview with series_roles requires series_limit >= 1, "
            f"got {series_limit!r}"
        )
    if allowed_series_keys is not None:
        if not series_roles:
            raise MaterialisationUsageError("allowed_series_keys requires at least one series role")
        undeclared_roles = set(series_roles).difference(request.declared_roles)
        if undeclared_roles:
            raise MaterialisationUsageError(
                f"series_roles must be declared on the request, got {sorted(undeclared_roles)!r}"
            )
        key_length = len(series_roles)
        if any(len(key) != key_length for key in allowed_series_keys):
            raise MaterialisationUsageError(
                f"allowed_series_keys must match the number of series_roles, got {key_length} roles"
            )
    prepared = _prepare_preview_member(
        request,
        sdmx=sdmx,
        graph=graph,
        limit=limit,
        ceiling=ceiling,
        rate_cache=rate_cache.for_request() if rate_cache is not None else None,
        coverage_index=coverage,
        deadline=deadline,
        precomputed=precomputed,
        series_roles=series_roles,
        series_limit=series_limit,
        allowed_series_keys=allowed_series_keys,
        elect=lambda _omitted: elected_year,
    )
    return PreparedPreview(prepared)


def finish_materialise_preview(
    prepared: PreparedPreview, *, time_period: str | None = None
) -> PreviewResult:
    """Convert and render a prepared preview at one exact period, or with none given at
    its elected period when it holds a row there, else at every period it holds."""
    outcome = _finish_preview_member(prepared._prepared, time_period=time_period)
    return _preview_result(outcome, prepared._prepared.request)


def materialise_preview(
    request: MaterialisationRequest,
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    limit: int,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    with_content_version: bool = False,
    rate_cache: RateTableCache | None = None,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
    precomputed: Sequence[Facet] | None = None,
    series_roles: Sequence[str] = (),
    series_limit: int | None = None,
    refuse_when_every_row_omitted: bool = True,
) -> PreviewResult:
    """Generate a write-free, single-member preview through the mint's facet selection and
    dispatch. The preview reads every row a mint reads and reports `truncated` and the full
    `total_rows`. `limit` bounds only the rows it delivers. `ceiling` sizes the raw-fetch guard
    as it does for a mint, 20 times `ceiling` raw rows, and a read past the guard raises
    `SdmxCapExceededError`. Every survivor is counted, which a caller whose `ceiling` equals
    its `limit` needs to truncate.

    Selection errors (`DefaultFacetUndefinedError`, `FacetNotAvailableError`),
    role-divergence errors and upstream failures match `materialise_spec`. Invalid input, including
    `limit < 1`, raises `MaterialisationUsageError`, a caller bug that must not fail open.
    Request-form units require `rate_cache` before fetching. Ref and unbound bindings skip
    caching. An absolute monotonic `deadline` bounds the read. An omitted deadline leaves the SDMX
    transport timeout as the only bound.

    With `with_content_version`, the preview's `content_version` matches a mint for the same
    request, including converted requests and any `limit`. Without it, `content_version` is
    `None`. Role-bearing requests use `coverage` for
    the mint's facet counts. `precomputed` hands over candidates the caller already probed, which
    skips the probe (`facets.probe_facets`); `request.candidate_filter` still narrows them.

    `series_roles` names roles the request leaves open whose every dcid is one series of the
    caller's: a `latest` window keeps each series' own latest period and elects no year, where
    a mint keeps the member's one, and conversion records each series
    (`PreviewResult.series_conversion_records`).
    Each series keeps its own most recent `series_limit` rows, which such a read requires, and
    `PreviewResult.series_total_rows` counts each series before the cut. `ceiling` sizes the raw
    fetch guard, so the caller sizes it for the whole read.

    A conversion that leaves no row, having omitted some entity's rows, refuses
    `currency_unmapped` or `deflator_missing` as a mint does (`refuse_if_every_row_omitted`).
    A caller previewing one part of a request passes `refuse_when_every_row_omitted=False`,
    gets the part's empty result with the omissions on its conversion records, and
    decides for the whole request.
    """
    if limit < 1:
        raise MaterialisationUsageError(f"materialise_preview requires limit >= 1, got {limit!r}")
    if series_roles and (series_limit is None or series_limit < 1):
        raise MaterialisationUsageError(
            f"materialise_preview with series_roles requires series_limit >= 1, "
            f"got {series_limit!r}"
        )

    def _generate() -> _PreviewOutcome:
        prepared = _prepare_preview_member(
            request,
            sdmx=sdmx,
            graph=graph,
            limit=limit,
            ceiling=ceiling,
            rate_cache=rate_cache.for_request() if rate_cache is not None else None,
            coverage_index=coverage,
            deadline=deadline,
            mint_full_frame=with_content_version,
            precomputed=precomputed,
            series_roles=series_roles,
            series_limit=series_limit,
        )
        return _finish_preview_member(prepared)

    outcome = _generate()
    if refuse_when_every_row_omitted:
        refuse_if_every_row_omitted(
            served_rows=outcome.coverage.rows, records=(outcome.conversion_record,)
        )
    return _preview_result(outcome, request)


def _preview_result(outcome: _PreviewOutcome, request: MaterialisationRequest) -> PreviewResult:
    """Build a preview result from one generation. With `outcome.full_frame`, mint
    `content_version` over the full survivor frame, including rows beyond the delivered limit.
    Without it, `content_version` is `None`.
    """
    full_frame = outcome.full_frame
    content_version = mint_version(full_frame) if full_frame is not None else None
    return PreviewResult(
        frame=outcome.frame,
        coverage=outcome.coverage,
        total_rows=outcome.total_rows,
        truncated=outcome.truncated,
        selections=outcome.selections,
        warnings=_build_warnings(request, outcome),
        conversion_record=outcome.conversion_record,
        conversion_recipe_id=outcome.conversion_recipe_id,
        content_version=content_version,
        default_scope=outcome.default_scope.places,
        series_conversion_records=outcome.series_conversion_records,
        series_total_rows=outcome.series_total_rows,
    )


def _check_group_role_agreement(requests: Sequence[MaterialisationRequest]) -> None:
    """Raise `MaterialisationUsageError` before network access when members declare different role
    sets.
    """
    role_sets = {frozenset(request.declared_roles) for request in requests}
    if len(role_sets) > 1:
        raise MaterialisationUsageError(
            "materialise_group got members with disagreeing declared_roles: "
            f"{sorted(tuple(sorted(roles)) for roles in role_sets)!r}"
        )


def _check_group_stat_var_uniqueness(requests: Sequence[MaterialisationRequest]) -> None:
    """Raise `MaterialisationUsageError` before network access for duplicate StatVars. Group
    observation counts use `stat_var` as their key, so duplicates would attach one member's count to
    both results.
    """
    seen: set[str] = set()
    for request in requests:
        if request.stat_var in seen:
            raise MaterialisationUsageError(
                f"materialise_group got two members sharing stat_var {request.stat_var!r}"
            )
        seen.add(request.stat_var)


def _refuse_group_national_currency(requests: Sequence[MaterialisationRequest]) -> None:
    """Raise `FacetBindingUnsupportedError` before network access when a group of several
    members binds its unit to each place's national currency: group selection elects one
    unit across the members, and the binding serves one subject-only variable."""
    unit = requests[0].facet_bindings.get("unit")
    if len(requests) > 1 and unit is not None and unit.state == "national_currency":
        raise FacetBindingUnsupportedError(
            f"a national-currency unit binding serves one variable; this definition resolves "
            f"to {len(requests)}",
            facet_pointer="/facets/unit",
        )


def _check_group_unit_binding_agreement(requests: Sequence[MaterialisationRequest]) -> None:
    """Raise `MaterialisationUsageError` before network access when members bind different unit
    values. The shared facet pool uses the first request's unit preference, so members must agree.
    A missing unit binding counts as unbound.

    Other facet axes narrow per member. A requested provenance or measurement method absent from a
    member's candidates is a graph-state refusal with that member's alternatives. A unit mismatch is
    a caller error.
    """
    first = requests[0].facet_bindings.get("unit", FacetBinding(state="unbound"))
    for other in requests[1:]:
        binding = other.facet_bindings.get("unit", FacetBinding(state="unbound"))
        if binding != first:
            raise MaterialisationUsageError(
                "materialise_group got members binding different values on facet_bindings "
                f"axis 'unit': {first!r} != {binding!r}"
            )
