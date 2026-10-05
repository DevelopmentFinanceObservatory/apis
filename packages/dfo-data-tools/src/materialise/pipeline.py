"""Materialisation entry points, result types and request validation.

Mints generate data under a revision snapshot and finish after the snapshot is confirmed. Previews
can use the same drift check and content version when given a revision reader.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from dc_sdmx import EntityFilter, SdmxClient
from dfo_platform.executor import submit_with_context
from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS
from materialise.codes import MaterialisationWarning
from materialise.content_version import MintPreimage, mint_collection_version
from materialise.convert import ConversionRecord
from materialise.enrichment import EnrichmentLookup
from materialise.errors import MaterialisationUsageError, OverCeilingError
from materialise.facets import (
    CollectionFacetProbe,
    ElectedDataset,
    FacetBinding,
    FacetSelection,
    admitted_candidates,
    probe_collection_facets,
    probe_facets,
)
from materialise.finish import (
    _build_warnings,
    _finish_group,
    _finish_member,
    _mint_group_content_version,
    _mint_member_content_version,
)
from materialise.frame_build import Frame
from materialise.generation import (
    PreviewBudget,
    _bare_stat_var_ref,
    _finish_member_fetch,
    _finish_preview_member,
    _flatten_bound_entities,
    _generate_chunk,
    _generate_group,
    _generate_member,
    _GroupOutcome,
    _MemberFetchPlan,
    _MemberFetchResult,
    _MemberOutcome,
    _plan_collection_chunks,
    _plan_member_fetch,
    _planning_graph,
    _prepare_preview_member,
    _PreparedPreview,
    _PreviewOutcome,
    _RawRowBudget,
    _select_member_facets,
    _ShapeChunk,
    dispatch_lanes,
)
from materialise.graph import Facet, GraphReader
from materialise.member_conversion import decide_member_conversion, refuse_if_every_row_omitted
from materialise.profile import OutputProfile
from materialise.progress import RowProgress
from materialise.rate_tables import RateTableCache
from materialise.reshape import TableLike
from materialise.revision import RevisionReader, RevisionSnapshot, read_generation
from materialise.series_conversion import SeriesConversion, SeriesKey
from materialise.statistics import SeriesStatisticsFile
from materialise.trace import ManifestTrace, MemberFamily
from materialise.window import Coverage, RequestedWindow
from origin_catalogue import note_un_unserved

if TYPE_CHECKING:
    # Keep the optional coverage index lazy because it imports NumPy, absent from pure installs.
    from series_coverage.index import CoverageIndex

# Deployment code supplies its configured bound; this is the package fallback.
DEFAULT_MEMBER_CONCURRENCY = 4

# A frame's period column, and the scratch column `_latest_rows` orders kept rows by.
_TIME_PERIOD_COLUMN = "time_period"
_ROW_INDEX_COLUMN = "_row_index"


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
    # are known together, so `__post_init__` is where a `state="request"`
    # binding stored under any key other than `"unit"` is caught -- a
    # `FacetBinding` itself carries no axis of its own.
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
    # A caller's facet filter over each probed candidate's provenance, method and unit. The
    # member elects only among the candidates it admits (`facets.admitted_candidates`), so
    # a filtered request serves its best admitted facet. None admits every candidate.
    candidate_filter: Callable[[Facet], bool] | None = None

    def __post_init__(self) -> None:
        for axis, binding in self.facet_bindings.items():
            if binding.state == "request" and axis != "unit":
                raise MaterialisationUsageError(
                    f"a request-state facet binding only applies to the unit axis, not {axis!r}"
                )


@dataclass(frozen=True)
class MaterialisationResult:
    """`frame_chunks` returns a fresh iterator of validated frames in increasing `row_sort_key`
    order. Group chunks read from Parquet, so a caller needing two passes calls it twice and peak
    memory stays at one chunk. `mint_preimage` is the exact input to `content_version`. The wider
    `trace.facets` wire shape differs from the preimage, so never rebuild it from `trace`.
    `conversion_record` is set for a single member and `None` for a group or an unconverted result.
    `member_conversion_records` maps each StatVar to its record or `None`.
    `series_conversions` holds one record per converted series, keyed by bare StatVar and
    role tuple (`materialise.series_conversion`), and is empty for an unconverted result.
    `default_scope` is the places the default place scope resolved when a request binds no
    entity (`materialise.place_scope`), and empty otherwise.
    """

    frame_chunks: Callable[[], Iterator[Frame]]
    content_version: str
    trace: ManifestTrace
    coverage: Coverage
    warnings: tuple[MaterialisationWarning, ...]
    series_statistics: SeriesStatisticsFile
    output_profile: OutputProfile
    conversion_record: ConversionRecord | None
    member_conversion_records: Mapping[str, ConversionRecord | None]
    mint_preimage: MintPreimage
    # Time spent computing the output profile, in seconds.
    output_profile_seconds: float = 0.0
    default_scope: tuple[str, ...] = ()
    series_conversions: Mapping[SeriesKey, SeriesConversion] = field(default_factory=dict)


@dataclass(frozen=True)
class ConcatenatedFrameChunks:
    """In-memory `TableLike` view that preserves row order. `render_and_store` uses it for no-op
    groupby without a pivot. Streaming groupby and pivot fold over chunks. Collections build one
    table per member.
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

    With a revision reader, generation runs under the mint's drift check and `content_version`
    matches a mint for the same request. Without one, the version is `None`. The version covers the
    full survivor set regardless of `limit`.

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
    def selections(self) -> Mapping[str, FacetSelection]:
        """Selected facets, for a caller validating its planned dispatch group."""
        return self._prepared.selections


@dataclass(frozen=True)
class CollectionMaterialisationResult:
    """Per-member results plus the shared generation's revision snapshot and
    the collection-level `content_version`. `materialise.finish.collection_trace` merges the
    member traces into the collection's one `ManifestTrace`.
    `default_scope` is the places the default place scope resolved for the
    members that bind no entity, in first-seen order, and empty otherwise.
    """

    members: tuple[tuple[str, MaterialisationResult], ...]
    content_version: str
    revision_snapshot: RevisionSnapshot
    default_scope: tuple[str, ...] = ()


def materialise_group(
    requests: Sequence[MaterialisationRequest],
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    revision_reader: RevisionReader,
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
) -> MaterialisationResult:
    """Generate and finish a group under one revision-drift check. Returns one frame, trace and
    content version covering every member.

    Before any network call, raises `MaterialisationUsageError` for an empty request, duplicate
    StatVars, different declared-role sets or conflicting unit bindings. Other facet-axis
    mismatches are graph-state refusals when a member lacks the requested facet.

    One request uses the member path. Multiple requests select a shared facet, group members with
    the same entity scope and `latest` status, and fetch each chunk with one SDMX call. At most
    `member_concurrency` chunks run at once. The aggregate row ceiling applies to the group.

    `precomputed_probe` must match requests in order. Its revision becomes the generation's
    starting snapshot. The probe is reused on the first attempt, and retries probe again. What
    its UN reads could not serve joins the caller's `recording_un_blocked_places` record. Groups
    above the probe-pair limit require a supplied probe. `deadline` is one absolute monotonic bound
    shared by every SDMX stream and retry.

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
    """
    if not requests:
        raise MaterialisationUsageError("materialise_group requires at least one member")
    _check_group_role_agreement(requests)
    _check_group_stat_var_uniqueness(requests)
    _check_group_unit_binding_agreement(requests)
    rate_cache = rate_cache.for_request() if rate_cache is not None else None
    if precomputed_probe is not None:
        note_un_unserved(precomputed_probe.un_unserved)

    if len(requests) == 1:
        first_probe: list[CollectionFacetProbe | None] = [precomputed_probe]

        def _generate_one() -> _MemberOutcome:
            probe = first_probe[0]
            first_probe[0] = None
            return _generate_member(
                requests[0],
                sdmx=sdmx,
                graph=graph,
                ceiling=ceiling,
                rate_cache=rate_cache,
                coverage=coverage,
                precomputed_probe=probe,
                deadline=deadline,
                progress=progress,
            )

        outcome, revision_snapshot = read_generation(
            revision_reader,
            _generate_one,
            before=precomputed_probe.revision if precomputed_probe is not None else None,
        )
        refuse_if_every_row_omitted(
            served_rows=outcome.coverage.rows, records=(outcome.conversion_record,)
        )
        return _finish_member(
            requests[0],
            outcome,
            revision_snapshot=revision_snapshot,
            materialiser_build=materialiser_build,
            capability_version=capability_version,
            enrichment=enrichment,
        )

    first_group_probe: list[CollectionFacetProbe | None] = [precomputed_probe]

    def _generate_grp() -> _GroupOutcome:
        probe = first_group_probe[0]
        first_group_probe[0] = None
        return _generate_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            ceiling=ceiling,
            member_concurrency=member_concurrency,
            rate_cache=rate_cache,
            coverage_index=coverage,
            precomputed_probe=probe,
            deadline=deadline,
            progress=progress,
        )

    outcome, revision_snapshot = read_generation(
        revision_reader,
        _generate_grp,
        before=precomputed_probe.revision if precomputed_probe is not None else None,
    )
    refuse_if_every_row_omitted(
        served_rows=outcome.coverage.rows, records=outcome.member_conversion_records.values()
    )
    return _finish_group(
        requests,
        outcome,
        revision_snapshot=revision_snapshot,
        materialiser_build=materialiser_build,
        capability_version=capability_version,
        enrichment=enrichment,
    )


def materialise_spec(
    request: MaterialisationRequest,
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    revision_reader: RevisionReader,
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
        revision_reader=revision_reader,
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
    deadline: float | None = None,
) -> Mapping[str, FacetSelection]:
    """Return the facets and conversion decision a mint would use. The function reads facet
    candidates but stops before observations and rate-table loading. `precomputed` skips the
    probe and ranks with `breadth`, each facet's series count under the request's window, as
    the probe's own count ranks a probed decision (`facets.MemberProbe.breadth_in`). `settle`
    goes to the probe (`facets.probe_facets`), and Search passes `False`.

    Role-bearing requests require `coverage` or raise `MaterialisationUsageError` before probing.
    Selection can raise `DefaultFacetUndefinedError` or `FacetNotAvailableError`. Request-form
    units can raise `RateCoverageGapError`, `NonCurrencyUnitError` or `RecipeNotDeclaredError`.
    Ref and unbound bindings return selected facets only.
    """
    if precomputed is None:
        probe = probe_facets(
            stat_var=_bare_stat_var_ref(request.stat_var),
            bound_entities=_flatten_bound_entities(request.role_bindings),
            graph=graph,
            availability=sdmx,
            settle=settle,
            declared_roles=request.declared_roles,
            role_bindings=request.role_bindings,
            coverage=coverage,
            deadline=deadline,
        )
        probed, breadth = probe.candidates, probe.breadth_in(request.window)
    else:
        # A decision fetches no rows, so precomputed candidates need no admission.
        probed = precomputed
    candidates = admitted_candidates(probed, request.candidate_filter)
    selections = _select_member_facets(request, candidates, graph=graph, breadth=breadth)
    decide_member_conversion(request=request, selections=selections, graph=graph, deadline=deadline)
    return selections


def _plan_collection_members(
    member_requests: Sequence[MaterialisationRequest],
    probe: CollectionFacetProbe,
    *,
    graph: GraphReader,
    sdmx: SdmxClient,
    rate_cache: RateTableCache | None,
    deadline: float | None,
) -> list[_MemberFetchPlan]:
    """Resolve each member's facet selection and conversion sequentially in request order. The first
    error is raised before that member's or any later member's SDMX fetch. Selection is local.
    Request-form unit resolution can load rate or deflator tables eagerly, and reads its graph
    arcs through one `_planning_graph` memo shared by every member.
    """
    graph = _planning_graph(member_requests, graph=graph)
    return [
        _plan_member_fetch(
            member_requests[i],
            probe.members[i],
            graph=graph,
            sdmx=sdmx,
            rate_cache=rate_cache,
            deadline=deadline,
        )
        for i in range(len(member_requests))
    ]


class _CollectionChunkPool:
    """Fetch collection members in chunks that share an entity scope. Members with different role
    bindings remain separate. Each chunk uses one SDMX request for its StatVars, then splits rows by
    `variableMeasured`.

    The sliding window runs at most `member_concurrency` chunks and consumes futures in request
    order. One `_RawRowBudget` bounds the entire call to `_RAW_FETCH_ROWS_GUARD_MULTIPLIER *
    ceiling` raw rows, including concurrent fetches. After `run()`, `outcomes_by_index` contains
    each member's outcome by request position.
    """

    def __init__(
        self,
        requests: Sequence[tuple[str, MaterialisationRequest]],
        plans: Sequence[_MemberFetchPlan],
        chunks: Sequence[_ShapeChunk],
        *,
        sdmx: SdmxClient,
        graph: GraphReader,
        ceiling: int,
        raw_budget: _RawRowBudget,
        coverage: CoverageIndex | None,
        deadline: float | None,
        member_concurrency: int,
        progress: RowProgress | None,
    ) -> None:
        self._requests = requests
        self._member_requests = [member_request for _, member_request in requests]
        self._plans = plans
        self._chunks = chunks
        self._sdmx = sdmx
        self._graph = graph
        self._ceiling = ceiling
        self._raw_budget = raw_budget
        self._coverage = coverage
        self._deadline = deadline
        self._member_concurrency = member_concurrency
        self._progress = progress
        self.outcomes_by_index: dict[int, _MemberOutcome] = {}
        self._cumulative_rows = 0
        self._next_ready = 0

    def _submit_chunk(
        self, pool: ThreadPoolExecutor, chunk: _ShapeChunk
    ) -> Future[dict[int, _MemberFetchResult]]:
        return submit_with_context(
            pool,
            _generate_chunk,
            chunk,
            self._requests,
            self._plans,
            sdmx=self._sdmx,
            graph=self._graph,
            ceiling=self._ceiling,
            raw_budget=self._raw_budget,
            coverage=self._coverage,
            deadline=self._deadline,
        )

    def _check_ready_prefix(self) -> None:
        """Count completed members in request order, stopping at the first pending index. Raise
        `OverCeilingError` as soon as the contiguous prefix exceeds `ceiling`.
        """
        while self._next_ready in self.outcomes_by_index:
            self._cumulative_rows += self.outcomes_by_index[self._next_ready].coverage.rows
            self._next_ready += 1
            if self._cumulative_rows > self._ceiling:
                raise OverCeilingError(
                    f"collection's aggregate materialised coverage of "
                    f"{self._cumulative_rows} rows across {self._next_ready} member(s) "
                    f"exceeds the sync materialisation ceiling of {self._ceiling} rows",
                    detail=(
                        f"collection's aggregate materialised coverage of "
                        f"{self._cumulative_rows} rows across {self._next_ready} "
                        f"member(s) exceeds the sync materialisation ceiling of "
                        f"{self._ceiling} rows"
                    ),
                    ceiling_kind="rows",
                    count=self._cumulative_rows,
                    cap=self._ceiling,
                )

    def _absorb_chunk_result(self, fetch_results: dict[int, _MemberFetchResult]) -> None:
        """Convert raw fetch results to `_MemberOutcome`s, record them by request index, update
        progress and check the aggregate ceiling. Raw tables become unreachable after conversion.
        """
        for member_index, fetch_result in fetch_results.items():
            outcome = _finish_member_fetch(
                self._member_requests[member_index],
                self._plans[member_index],
                fetch_result,
                ceiling=self._ceiling,
            )
            self.outcomes_by_index[member_index] = outcome
            if self._progress is not None:
                self._progress.add(outcome.coverage.rows)
        self._check_ready_prefix()

    def run(self) -> list[tuple[str, _MemberOutcome]]:
        """Run chunks and return outcomes in request order. Futures are consumed in submission
        order, so the first reported failure follows request order. A ceiling crossing stops further
        submissions. Shutdown waits for submitted fetches to finish, preventing work from escaping
        this call or overlapping a revision retry.
        """
        chunks = self._chunks
        max_workers = max(1, min(self._member_concurrency, len(chunks)))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            pending = [
                self._submit_chunk(pool, chunks[i]) for i in range(min(max_workers, len(chunks)))
            ]
            next_chunk = len(pending)
            for _chunk in chunks:
                self._absorb_chunk_result(pending.pop(0).result())
                if next_chunk < len(chunks):
                    pending.append(self._submit_chunk(pool, chunks[next_chunk]))
                    next_chunk += 1
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        return [
            (spec_id, self.outcomes_by_index[i])
            for i, (spec_id, _member_request) in enumerate(self._requests)
        ]


def materialise_collection(
    requests: Sequence[tuple[str, MaterialisationRequest]],
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    revision_reader: RevisionReader,
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
) -> CollectionMaterialisationResult:
    """Materialise every `(spec_id, request)` under one revision check and mint one order-invariant
    collection content version. Duplicate `spec_id`s raise `MaterialisationUsageError` before any
    network call.

    The collection probe runs once per attempt. `precomputed_probe` must match request order, and
    its revision is the generation's starting snapshot. Retries probe again. What a probe's UN
    reads could not serve joins the caller's `recording_un_blocked_places` record. Facet
    selection and conversion resolve in request order before SDMX fetching through a bounded
    chunk pool. `ceiling` limits aggregate rows across all members.

    Request-form unit bindings use `rate_cache`. A missing cache raises `MaterialisationUsageError`
    before fetching. Ref and unbound bindings skip the cache. Role-bearing requests require
    `coverage` for facet probes and SDMX dispatch. One absolute `deadline` covers all chunks and
    waves.
    `progress` increments by each completed member's row count. The caller supplies
    `materialiser_build` and `capability_version`.

    Every member converts against one version of each rate table, because the call loads tables
    through a `RateTableCache.for_request` view. A member whose rows a conversion all omitted,
    for entities no single currency names or for deflators the table lacks, serves no row;
    the collection refuses `currency_unmapped` or `deflator_missing` only when no member
    serves a row (`refuse_if_every_row_omitted`).
    """
    _reject_duplicate_spec_ids(requests)
    rate_cache = rate_cache.for_request() if rate_cache is not None else None

    first_probe: list[CollectionFacetProbe | None] = [precomputed_probe]

    def _generate_all() -> tuple[list[tuple[str, _MemberOutcome]], tuple[str, ...]]:
        member_requests = [member_request for _, member_request in requests]
        probe = first_probe[0]
        first_probe[0] = None
        if probe is None:
            # Only a supplied probe provides `read_generation`'s `before` stamp. An in-closure
            # probe runs between its before and after reads.
            probe = probe_collection_facets(
                member_requests,
                graph=graph,
                availability=sdmx,
                settle=True,
                coverage=coverage,
                deadline=deadline,
            )
        note_un_unserved(probe.un_unserved)
        plans = _plan_collection_members(
            member_requests, probe, graph=graph, sdmx=sdmx, rate_cache=rate_cache, deadline=deadline
        )
        # Share lane calculation with group generation and admission estimates.
        # Distinct entity scopes form singleton chunks. Shared scopes can combine.
        chunks = _plan_collection_chunks(
            requests,
            places=[member.places for member in probe.members],
            lanes=dispatch_lanes(member_concurrency, len(requests)),
        )
        pool_runner = _CollectionChunkPool(
            requests,
            plans,
            chunks,
            sdmx=sdmx,
            graph=graph,
            ceiling=ceiling,
            raw_budget=_RawRowBudget(ceiling),
            coverage=coverage,
            deadline=deadline,
            member_concurrency=member_concurrency,
            progress=progress,
        )
        return pool_runner.run(), probe.default_scope

    (outcomes, default_scope), revision_snapshot = read_generation(
        revision_reader,
        _generate_all,
        before=precomputed_probe.revision if precomputed_probe is not None else None,
    )
    refuse_if_every_row_omitted(
        served_rows=sum(outcome.coverage.rows for _, outcome in outcomes),
        records=(outcome.conversion_record for _, outcome in outcomes),
    )

    members = tuple(
        (
            spec_id,
            _finish_member(
                member_request,
                outcome,
                revision_snapshot=revision_snapshot,
                materialiser_build=materialiser_build,
                capability_version=capability_version,
                enrichment=enrichment,
            ),
        )
        for (spec_id, member_request), (_, outcome) in zip(requests, outcomes, strict=True)
    )
    collection_content_version = mint_collection_version(
        [(spec_id, result.content_version) for spec_id, result in members]
    )
    return CollectionMaterialisationResult(
        members=members,
        content_version=collection_content_version,
        revision_snapshot=revision_snapshot,
        default_scope=default_scope,
    )


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
    refuse_over_ceiling: bool = False,
    budget: PreviewBudget | None = None,
    precomputed: Sequence[Facet] | None = None,
    series_roles: Sequence[str] = (),
    series_limit: int | None = None,
    allowed_series_keys: frozenset[tuple[str, ...]] | None = None,
) -> PreparedPreview:
    """Read and reduce a preview without applying conversion arithmetic.

    ``finish_materialise_preview`` may select one exact period before it
    converts.  This is intended for coordinated Search latest reads.
    """
    if limit < 1:
        raise MaterialisationUsageError(f"materialise_preview requires limit >= 1, got {limit!r}")
    if budget is not None and (not refuse_over_ceiling or budget.ceiling != ceiling):
        raise MaterialisationUsageError(
            "materialise_preview with a budget requires refuse_over_ceiling and the budget's "
            f"ceiling {budget.ceiling!r}, got refuse_over_ceiling={refuse_over_ceiling!r}, "
            f"ceiling={ceiling!r}"
        )
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
        refuse_over_ceiling=refuse_over_ceiling,
        budget=budget,
        precomputed=precomputed,
        series_roles=series_roles,
        series_limit=series_limit,
        allowed_series_keys=allowed_series_keys,
    )
    return PreparedPreview(prepared)


def finish_materialise_preview(
    prepared: PreparedPreview, *, time_period: str | None = None
) -> PreviewResult:
    """Convert and render a prepared preview, optionally at one exact period."""
    outcome = _finish_preview_member(prepared._prepared, time_period=time_period)
    if prepared._prepared.budget is not None:
        prepared._prepared.budget.commit(raw_rows=outcome.raw_rows, survivors=outcome.total_rows)
    return _preview_result(outcome, prepared._prepared.request, revision_snapshot=None)


def materialise_preview(
    request: MaterialisationRequest,
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    limit: int,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    revision_reader: RevisionReader | None = None,
    rate_cache: RateTableCache | None = None,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
    refuse_over_ceiling: bool = False,
    budget: PreviewBudget | None = None,
    precomputed: Sequence[Facet] | None = None,
    series_roles: Sequence[str] = (),
    series_limit: int | None = None,
    refuse_when_every_row_omitted: bool = True,
) -> PreviewResult:
    """Generate a write-free, single-member preview through the mint's facet selection and
    dispatch. The preview reads every row a mint reads and reports `truncated` and the full
    `total_rows`. `limit` bounds only the rows it delivers. `ceiling` sizes the raw-fetch guard
    as it does for a mint, 20 times `ceiling` raw rows, and a read past the guard raises
    `SdmxCapExceededError`. With `refuse_over_ceiling`, served rows past `ceiling` raise the
    mint's `OverCeilingError` at the point the mint's check runs, mid-read where it can, so the
    preview refuses where a mint refuses and holds no more rows. Without it, every survivor is counted, which a caller whose `ceiling` equals
    its `limit` needs to truncate. `materialise_group_preview` previews several members as one
    group.

    `budget` previews this member as one of a request's members: their raw reads draw on one
    `20 * ceiling` total and their survivors together refuse past `ceiling`, as a group or
    collection mint's do. It needs `refuse_over_ceiling` and the `ceiling` it was built with, or
    `MaterialisationUsageError` is raised. A member that succeeds joins the budget's totals.

    Selection errors (`DefaultFacetUndefinedError`, `FacetNotAvailableError`),
    role-divergence errors and upstream failures match `materialise_spec`. Invalid input, including
    `limit < 1`, raises `MaterialisationUsageError`, a caller bug that must not fail open.
    Request-form units require `rate_cache` before fetching. Ref and unbound bindings skip
    caching. An absolute monotonic `deadline` bounds the read. An omitted deadline leaves the SDMX
    transport timeout as the only bound.

    With `revision_reader`, preview generation uses the mint's drift check and its
    `content_version` matches a mint for the same request, including converted requests and any
    `limit`. Without a reader, `content_version` is `None`. Role-bearing requests use `coverage` for
    the mint's facet counts. `precomputed` hands over candidates the caller already probed, which
    skips the probe (`facets.probe_facets`); `request.candidate_filter` still narrows them.

    `series_roles` names roles the request leaves open whose every dcid is one series of the
    caller's: a `latest` window keeps each series' own latest period, where a mint keeps the
    member's one, and conversion records each series (`PreviewResult.series_conversion_records`).
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
    if budget is not None and (not refuse_over_ceiling or budget.ceiling != ceiling):
        raise MaterialisationUsageError(
            "materialise_preview with a budget requires refuse_over_ceiling and the budget's "
            f"ceiling {budget.ceiling!r}, got refuse_over_ceiling={refuse_over_ceiling!r}, "
            f"ceiling={ceiling!r}"
        )
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
            mint_full_frame=revision_reader is not None,
            refuse_over_ceiling=refuse_over_ceiling,
            budget=budget,
            precomputed=precomputed,
            series_roles=series_roles,
            series_limit=series_limit,
        )
        return _finish_preview_member(prepared)

    if revision_reader is None:
        outcome, revision_snapshot = _generate(), None
    else:
        outcome, revision_snapshot = read_generation(revision_reader, _generate)
    if refuse_when_every_row_omitted:
        refuse_if_every_row_omitted(
            served_rows=outcome.coverage.rows, records=(outcome.conversion_record,)
        )
    if budget is not None:
        budget.commit(raw_rows=outcome.raw_rows, survivors=outcome.total_rows)
    return _preview_result(outcome, request, revision_snapshot=revision_snapshot)


def _preview_result(
    outcome: _PreviewOutcome,
    request: MaterialisationRequest,
    *,
    revision_snapshot: RevisionSnapshot | None,
) -> PreviewResult:
    """Build a preview result from one generation. With a revision snapshot, mint `content_version`
    over the full survivor frame, including rows beyond the delivered limit. A missing snapshot
    yields `content_version=None`. A snapshot pairs with `outcome.full_frame`.
    """
    full_frame = outcome.full_frame
    if revision_snapshot is None:
        content_version = None
    else:
        assert full_frame is not None
        content_version = _mint_member_content_version(
            request,
            outcome.selections,
            outcome.conversion_record,
            full_frame,
            revision_snapshot=revision_snapshot,
        ).content_version
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


def materialise_group_preview(
    requests: Sequence[MaterialisationRequest],
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    revision_reader: RevisionReader,
    limit: int,
    ceiling: int = DEFAULT_SYNC_CEILING_ROWS,
    member_concurrency: int = DEFAULT_MEMBER_CONCURRENCY,
    rate_cache: RateTableCache | None = None,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
) -> PreviewResult:
    """Generate a write-free preview of the group `materialise_group` serves for `requests`.

    One request previews through `materialise_preview` with the mint's ceiling refusals.
    Several run the group's own generation: one facet selection shared by every member, the
    members fetched and pooled under `ceiling`, and one `content_version` minted as
    `materialise_group` mints it. `total_rows`, `coverage`, `content_version` and
    `selections` are therefore the group mint's, and every refusal of that mint's
    generation is this preview's.

    `frame` keeps the `limit` rows with the greatest `time_period`, null periods first to
    drop and a tie at the boundary period resolved in favour of the later rows, in the
    group frame's row order. `conversion_record` is `None`, as it is on a group result;
    `conversion_recipe_id` is the recipe the converted members share. Invalid input,
    including `limit < 1`, raises `MaterialisationUsageError`.

    Every member converts against one version of each rate table, because the call loads tables
    through a `RateTableCache.for_request` view.
    """
    if limit < 1:
        raise MaterialisationUsageError(
            f"materialise_group_preview requires limit >= 1, got {limit!r}"
        )
    if not requests:
        raise MaterialisationUsageError("materialise_group_preview requires at least one member")
    if len(requests) == 1:
        return materialise_preview(
            requests[0],
            sdmx=sdmx,
            graph=graph,
            limit=limit,
            ceiling=ceiling,
            revision_reader=revision_reader,
            rate_cache=rate_cache,
            coverage=coverage,
            deadline=deadline,
            refuse_over_ceiling=True,
        )
    _check_group_role_agreement(requests)
    _check_group_stat_var_uniqueness(requests)
    _check_group_unit_binding_agreement(requests)
    rate_cache = rate_cache.for_request() if rate_cache is not None else None

    def _generate() -> _GroupOutcome:
        return _generate_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            ceiling=ceiling,
            member_concurrency=member_concurrency,
            rate_cache=rate_cache,
            coverage_index=coverage,
            deadline=deadline,
        )

    outcome, revision_snapshot = read_generation(revision_reader, _generate)
    refuse_if_every_row_omitted(
        served_rows=outcome.coverage.rows, records=outcome.member_conversion_records.values()
    )
    mint = _mint_group_content_version(outcome, revision_snapshot=revision_snapshot)
    recipe_ids = [
        recipe_id
        for _stat_var, recipe_id in sorted(outcome.member_conversion_recipe_ids.items())
        if recipe_id is not None
    ]
    return PreviewResult(
        frame=_latest_rows(outcome.frame_chunks(), limit=limit),
        coverage=outcome.coverage,
        total_rows=outcome.row_count,
        truncated=outcome.row_count > limit,
        selections=outcome.selections,
        warnings=_build_warnings(requests[0], outcome),
        conversion_record=None,
        conversion_recipe_id=recipe_ids[0] if recipe_ids else None,
        content_version=mint.content_version,
        default_scope=outcome.default_scope.places,
    )


def _latest_rows(chunks: Iterator[Frame], *, limit: int) -> Frame:
    """Return the `limit` rows of `chunks` with the greatest `time_period`, in chunk order.

    Null periods sort first and a tie at the boundary period keeps the later rows. One chunk
    and at most `limit` kept rows are held at a time. `chunks` yields at least one frame.
    """
    columns: tuple[str, ...] = ()
    kept: pa.Table | None = None
    offset = 0
    for chunk in chunks:
        columns = chunk.columns
        indexed = chunk.table.append_column(
            _ROW_INDEX_COLUMN,
            pa.array(range(offset, offset + chunk.table.num_rows), type=pa.int64()),
        )
        offset += chunk.table.num_rows
        pooled = indexed if kept is None else pa.concat_tables([kept, indexed])
        if pooled.num_rows > limit:
            order = pc.sort_indices(
                pooled,
                sort_keys=[(_TIME_PERIOD_COLUMN, "ascending"), (_ROW_INDEX_COLUMN, "ascending")],
                null_placement="at_start",
            )
            pooled = pooled.take(order[-limit:])
        kept = pooled
    if kept is None:
        raise MaterialisationUsageError("a group preview needs at least one frame chunk")
    in_order = kept.take(pc.sort_indices(kept, sort_keys=[(_ROW_INDEX_COLUMN, "ascending")]))
    return Frame(columns=columns, table=in_order.drop_columns([_ROW_INDEX_COLUMN]))


def _reject_duplicate_spec_ids(requests: Sequence[tuple[str, MaterialisationRequest]]) -> None:
    """Raise `MaterialisationUsageError` on the first repeated `spec_id`
    before any network call.
    """
    seen: set[str] = set()
    for spec_id, _request in requests:
        if spec_id in seen:
            raise MaterialisationUsageError(
                f"materialise_collection got two members sharing spec_id {spec_id!r}"
            )
        seen.add(spec_id)


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
