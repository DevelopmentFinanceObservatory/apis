"""Re-resolve dataset parts through the shared region combiner.

Identity entries call `resolve_spec_resubmit` in worker threads, or take the region the
response cache's part memo stored for the same payload under the same engine options and
stamps (`PartMemoScope`). A hit binds its relaunch closures over this request's clients
(`promote_relaunch.bind_relaunches`). Candidate entries
promote each candidate under the same request semaphore. Placeholders build a
`RegionResult` without graph or LLM calls. Parsed entries use one shared fan-out.
Only that entry kind calls the LLM, once per entry to bind it.

The request semaphore caps identity/candidate promotes and parsed variable legs
together. Entity pre-resolution is one `resolver.resolve()` call, outside that cap.

Promote tasks start before parsed fan-out, so both tracks use the deadline in
parallel. Candidate regions carry the posted ordering override so the combiner
preserves candidate order.

The door emits no interpretation or per-leg progress events. Parsed fan-out
indices are local to its sub-list, so those events are drained to avoid labels
that mismatch `inp.entries`.

`EngineInputError` returns 4xx with pointers mapped to `/input/entries/{i}/...`.
Static validation of placeholder codes and candidate counts runs before tasks.
The lowest-index static error wins over resolution errors. Promote tasks all finish
before selection. The lowest-index `EngineInputError` beats infrastructure errors.
Otherwise the lowest-index exception propagates. Resolution outcomes remain 200 parts.

Timeouts and infrastructure errors fail the request with 504 or 5xx. Timeouts never
return placeholder no-data. On parsed-fan-out timeout, an already-completed promote
`EngineInputError` takes precedence. Pending promotes are ignored so the deadline
holds.

Identity and candidate promotes elect their members as the Datasets mint does. A member
walk whose first page truncates without proof reads the whole filtered set under a
deadline `resubmit_full_read_timeout_s` (90 s) from request start, and the promote's later
phases share it. The wait for promotes is bound by that deadline, since a promote's own
reads end at `params.deadline` (45 s) unless its walk ran the full read. The trailing
stream runs to the latest `RegionResult.reads_deadline` among the parts, so it keeps the
request deadline unless a promote ran the full read.

An identity entry that posts `scope_entries` compiles them over its identity
(`scope.build.build_part_scope`), resolves the identity they compile to, and carries the
scope on its definite part (`_compiled_scope`, `_served_scope`). A composite identity, posted
or compiled, resolves the identity its option was set on
(`identity_edit.composite_table_identity`) and answers as the composite (`_as_composite`).

A part whose answer stands on less than a later request may read
(`RegionResult.leg_degraded`) marks the `DoneEvent` `degraded_leg`, as the parsed door does.

Four task-creation sites register trailing handles with the request scope, whose
`core.resolve_events` cleanup cancels and awaits unclaimed tasks on exit. Parsed
fan-out also cleans up produced tasks on infrastructure errors and before raising
its timeout. Successful handles are consumed by the trailing streamer.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from contextlib import aclosing

from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.build import build_commit
from dfo_platform.named import cut
from dfo_platform.trace import get_trace_id
from entity_resolution import Admission, EntityResolver
from materialise import GraphReader
from materialise.member_conversion import request_conversion_reader
from materialise.rate_tables import RateTableCache
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.codes import CODE_CATALOG
from search.engine.concurrency import bounded_semaphore
from search.engine.config import settings
from search.engine.conjoin import assemble_region
from search.engine.deadline import _await_bounded
from search.engine.enrichment import _launch_trailing_tasks, _stream_slot_options
from search.engine.entry.dataset_entries import (
    check_candidates_ceiling,
    resolve_candidates_entry,
    resolve_parsed_entries,
    spec_resubmit_input_for_identity,
)
from search.engine.entry.multi import _stream_combined_regions
from search.engine.entry.params import EntryParams
from search.engine.envelope import _terminal_events, now_ms
from search.engine.errors import EngineInputError
from search.engine.events import DoneEvent, DoneTelemetry, Event, LateDepartures
from search.engine.identity_edit import composite_table_identity
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region import RegionResult, resolve_spec_resubmit
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.region.promote_relaunch import bind_relaunches
from search.engine.region.result import with_composite_answer, with_runnable_operations
from search.engine.region.sentence_events import answer_events, response_sentence_events
from search.engine.region.state import EngineOptions
from search.engine.response_cache_flow import PartMemoScope
from search.engine.retrieval.bound_values import identity_dataset
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    DatasetResubmitInput,
    EntityGroup,
    PartScope,
    QueryEcho,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
    ResubmitParsedEntry,
    ResubmitPlaceholderEntry,
    Unsupported,
    Warning,
)
from search.scope.basis import identity_basis_record
from search.scope.build import build_part_scope, refusal_error
from search.scope.compile import Refusal
from search.scope.composite import assemble_composite
from search.scope.region import region_groups
from search.scope.tables import PartFacts, ScopeFacts
from series_coverage.shared import CoverageSnapshot


def full_read_deadline_for(deadline: float) -> float:
    """The monotonic deadline of a resubmit whose member walk runs the full read, for a
    request whose own deadline is `deadline`: `resubmit_full_read_timeout_s` from request
    start where `deadline` is `request_timeout_s` from it."""
    return deadline + settings().resubmit_full_read_timeout_s - settings().request_timeout_s


def _compiled_scope(
    entry: ResubmitIdentityEntry, *, index: int, coverage: CoverageSnapshot
) -> PartScope:
    """The scope `entry`'s scope entries compile to over its identity, with the basis record
    the identity reads (`basis.identity_basis_record`). Raises the compile's refusal as a
    422 error pointed at the scope entry it refuses."""
    assert entry.scope_entries is not None
    dataset = identity_dataset(entry.identity)
    # The comprehension widens the posted role keys to `str`, which `dict()` cannot.
    groups: dict[str, EntityGroup] = {  # noqa: C416
        role: group for role, group in (entry.entity_groups or {}).items()
    }
    basis = identity_basis_record(entry.identity, get_index(), entry.scope_entries)
    built = build_part_scope(
        base=entry.identity,
        entries=entry.scope_entries,
        records={} if basis is None else {"measure_basis": basis},
        facts=ScopeFacts(
            index=get_index(),
            coverage=coverage,
            part=PartFacts(
                operations=tuple(entry.operations),
                entity_groups=groups,
                datasets=() if dataset is None else (dataset,),
            ),
        ),
    )
    if isinstance(built, Refusal):
        raise refusal_error(built, pointer=f"/input/entries/{index}/scope_entries/{built.entry}")
    return built


def _as_composite(
    region: RegionResult,
    identity: BaseIdentity,
    *,
    scope: PartScope | None,
    coverage: CoverageSnapshot,
) -> RegionResult:
    """`region`, resolved for composite `identity`'s table identity, answering the composite
    (`scope.composite.assemble_composite`) under its own spec_id, which is the compiled
    scope's when `scope` compiled it."""
    spec_id = scope.compiled.spec_id if scope is not None else identity_spec_id(identity)
    facts = ScopeFacts(index=get_index(), coverage=coverage)
    spec, _ = assemble_composite(region.spec, identity, spec_id, facts=facts)
    return with_composite_answer(region, spec)


def _served_scope(scope: PartScope | None, region: RegionResult) -> PartScope | None:
    """`scope`, for a region answering a Spec; None when it answers none."""
    return scope if region.status == "definite" else None


def _rewrite_pointer(exc: EngineInputError, index: int) -> EngineInputError:
    """Map a promote refusal to this entry's `/input/entries/{index}/...` pointer.

    Pointer-less or unrecognised pointers pass through unchanged.
    """
    pointer = exc.pointer
    if pointer is None or not pointer.startswith("/input"):
        return exc
    new_pointer = f"/input/entries/{index}" + pointer.removeprefix("/input")
    return EngineInputError(
        str(exc),
        code=exc.code,
        status=exc.status,
        pointer=new_pointer,
        alternatives=exc.alternatives,
    )


def _first_completed_refusal(
    promote_tasks: dict[int, asyncio.Task[RegionResult]],
) -> EngineInputError | None:
    """Return the lowest-index completed input refusal, or `None`.

    Used on parsed timeout. Only completed promotes are considered.
    `Task.exception()` raises `CancelledError` for a cancelled task.
    """
    refusals = {
        i: exc
        for i, t in promote_tasks.items()
        if t.done() and not t.cancelled() and isinstance(exc := t.exception(), EngineInputError)
    }
    if not refusals:
        return None
    return refusals[min(refusals)]


def _select_promote_outcome(
    promote_tasks: dict[int, asyncio.Task[RegionResult]],
) -> list[RegionResult]:
    """Select completed promote outcomes in entry order.

    The lowest-index `EngineInputError` wins. Otherwise the lowest-index exception
    propagates. With no failures, return results in entry order. Call after all
    tasks finish so reading `Task.exception()` cannot block.
    """
    refusal = _first_completed_refusal(promote_tasks)
    if refusal is not None:
        raise refusal
    for i in sorted(promote_tasks):
        task = promote_tasks[i]
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            raise exc
    return [promote_tasks[i].result() for i in promote_tasks]


def _unsupported_detail(code: str, *, index: int) -> str:
    """Return registry detail for a posted code. Refuse unknown codes as `request_invalid`."""
    entry = CODE_CATALOG.get(code)
    if entry is None:
        message = f"{cut(code)!r} is not a code this deployment's registry recognises"
        raise EngineInputError(
            message,
            code="request_invalid",
            pointer=f"/input/entries/{index}/code",
        )
    return entry[1]


def _part_restated_warning(entry: ResubmitPlaceholderEntry) -> Warning:
    return Warning(
        code="part_restated",
        severity="info",
        message=(
            f"'{entry.variable_text}' was restated from a posted {entry.status} "
            "outcome; not re-resolved."
        ),
    )


def _placeholder_region(entry: ResubmitPlaceholderEntry, index: int) -> RegionResult:
    """Build a no-data or unsupported region inline with zero graph and LLM calls."""
    warning = _part_restated_warning(entry)
    if entry.status == "no_data":
        assert entry.reason is not None, (
            "ResubmitPlaceholderEntry's own validator requires reason when status is no_data"
        )
        return RegionResult(
            variable_text=entry.variable_text,
            status="no_data",
            specs=(),
            no_data_reason=entry.reason,
            warnings=(warning,),
            timing_by_step={},
            earliest_index=index,
            separate=entry.separate,
        )
    assert entry.code is not None, (
        "ResubmitPlaceholderEntry's own validator requires code when status is unsupported"
    )
    return RegionResult(
        variable_text=entry.variable_text,
        status="unsupported",
        specs=(),
        no_data_reason=None,
        warnings=(warning,),
        timing_by_step={},
        earliest_index=index,
        separate=entry.separate,
        unsupported=Unsupported(
            code=entry.code,
            pointer="",
            detail=_unsupported_detail(entry.code, index=index),
            definition={},
        ),
    )


async def handle(
    inp: DatasetResubmitInput,
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    start_ms: int,
    params: EntryParams,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    memo: PartMemoScope | None = None,
) -> AsyncIterator[Event]:
    # Validate placeholder codes and candidate counts before creating tasks.
    # These static errors precede resolution failures and point to their entry.
    for i, entry in enumerate(inp.entries):
        if isinstance(entry, ResubmitPlaceholderEntry) and entry.status == "unsupported":
            assert entry.code is not None, (
                "ResubmitPlaceholderEntry's own validator requires code when status is unsupported"
            )
            _unsupported_detail(entry.code, index=i)
        elif isinstance(entry, ResubmitCandidatesEntry):
            check_candidates_ceiling(entry, index=i)

    # One request-scoped semaphore bounds promotes and parsed fan-out together.
    sem = bounded_semaphore()
    full_read_deadline = full_read_deadline_for(params.deadline)

    options = EngineOptions(
        include_option_coverage=params.include_option_coverage,
        stream_slot_options=params.stream_option_coverage,
        include_observations=params.include_observations,
        stream_observations=params.stream_observations,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        deadline=params.deadline,
    )

    async def _resolve_one(index: int, entry: ResubmitIdentityEntry) -> RegionResult:
        # A composite its components cannot serve refuses before its scope compiles.
        composite_table_identity(entry.identity, pointer=f"/input/entries/{index}/identity")
        # The scope build reads coverage and scans families, so it runs off the event loop.
        scope = (
            None
            if entry.scope_entries is None
            else await asyncio.to_thread(
                _compiled_scope, entry, index=index, coverage=params.coverage
            )
        )
        identity = entry.identity if scope is None else scope.compiled.identity
        read = composite_table_identity(identity, pointer=f"/input/entries/{index}/identity")
        # A compiled scope may switch the region's reading, so its groups are restated.
        groups = (
            entry.entity_groups
            if scope is None or entry.entity_groups is None
            else region_groups(entry.entity_groups, read, index=get_index())
        )
        payload = (
            spec_resubmit_input_for_identity(entry)
            if scope is None and read is identity
            else ResubmitIdentityPayload(identity=read, entity_groups=groups)
        )
        key = memo.key(payload, options) if memo is not None and memo.active else None
        stored = memo.lookup(key) if memo is not None and key is not None else None
        if stored is not None:
            region = _memo_hit(stored)
        else:
            async with sem:
                try:
                    region = await asyncio.to_thread(
                        resolve_spec_resubmit,
                        inp=payload,
                        graph=graph,
                        sdmx=sdmx,
                        coverage=params.coverage,
                        options=options,
                        full_read_deadline=full_read_deadline,
                    )
                except EngineInputError as exc:
                    raise _rewrite_pointer(exc, index) from exc
            # Stored before any trailing task launches, without its closures.
            if memo is not None and key is not None:
                memo.store(key, region)
        if read is not identity and region.status == "definite":
            region = _as_composite(region, identity, scope=scope, coverage=params.coverage)
        return dataclasses.replace(
            region,
            assumption_echo=None if entry.assumptions is None else tuple(entry.assumptions),
            variable_text=entry.variable_text,
            earliest_index=index,
            separate=entry.separate,
            scope_choices=_served_scope(scope, region),
        )

    def _memo_hit(stored: RegionResult) -> RegionResult:
        """Return `stored` with this request's relaunch closures, built over its own conversion
        memo as `resolve_spec_resubmit` builds them, and the lookup as its only timing step.
        A hit runs no member walk, so its trailing reads run to this request's deadline."""
        started = now_ms()
        hit_options = (
            dataclasses.replace(
                options, materialise_graph=request_conversion_reader(materialise_graph)
            )
            if materialise_graph is not None
            else options
        )
        region = bind_relaunches(
            stored, graph=graph, sdmx=sdmx, coverage=params.coverage, options=hit_options
        )
        return dataclasses.replace(
            region,
            timing_by_step={"response_cache_part_memo": now_ms() - started},
            reads_deadline=params.deadline,
        )

    # Start identity and candidate promotes before parsed fan-out. Placeholders
    # build regions inline and do not create promote tasks.
    promote_tasks: dict[int, asyncio.Task[RegionResult]] = {}
    for i, entry in enumerate(inp.entries):
        if isinstance(entry, ResubmitIdentityEntry):
            promote_tasks[i] = asyncio.create_task(_resolve_one(i, entry))
        elif isinstance(entry, ResubmitCandidatesEntry):
            promote_tasks[i] = asyncio.create_task(
                resolve_candidates_entry(
                    i,
                    entry,
                    graph=graph,
                    sdmx=sdmx,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=rate_cache,
                    population_memo=population_memo,
                    place_options_pool=place_options_pool,
                    deadline=params.deadline,
                    full_read_deadline=full_read_deadline,
                    sem=sem,
                    coverage=params.coverage,
                )
            )

    parsed_entries = [
        (i, entry) for i, entry in enumerate(inp.entries) if isinstance(entry, ResubmitParsedEntry)
    ]

    async def _gather_all() -> list[RegionResult]:
        """Wait for every promote before selecting an outcome by entry index."""
        await asyncio.gather(*promote_tasks.values(), return_exceptions=True)
        return _select_promote_outcome(promote_tasks)

    parsed_regions_by_index: dict[int, RegionResult] = {}
    try:
        try:
            # Parsed entries share one fan-out. Promote tasks keep running.
            if parsed_entries:
                parsed_regions_by_index = await resolve_parsed_entries(
                    parsed_entries,
                    graph=graph,
                    resolver=resolver,
                    admission=admission,
                    llm=llm,
                    binder=params.binder,
                    sdmx=sdmx,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    rate_cache=rate_cache,
                    population_memo=population_memo,
                    place_options_pool=place_options_pool,
                    pac=params.pac,
                    deadline=params.deadline,
                    include_option_coverage=params.include_option_coverage,
                    stream_slot_options=params.stream_option_coverage,
                    include_observations=params.include_observations,
                    stream_observations=params.stream_observations,
                    coverage=params.coverage,
                )
        except TimeoutError:
            # An already-completed promote refusal takes precedence. Pending
            # promotes remain unconsulted so the parsed deadline holds.
            refusal = _first_completed_refusal(promote_tasks)
            if refusal is not None:
                raise refusal from None
            raise

        # Timeout fails the request instead of replacing a resolved part with no-data. A
        # promote that ran no full read ends by its own reads' deadline, `params.deadline`.
        resolved = (
            await _await_bounded(_gather_all(), deadline=full_read_deadline)
            if promote_tasks
            else []
        )
    finally:
        for t in promote_tasks.values():
            if not t.done():
                t.cancel()
        if promote_tasks:
            await asyncio.gather(*promote_tasks.values(), return_exceptions=True)

    resolved_by_index = dict(zip(promote_tasks.keys(), resolved, strict=True))

    regions: list[RegionResult] = []
    for i, entry in enumerate(inp.entries):
        if isinstance(entry, ResubmitPlaceholderEntry):
            region = _placeholder_region(entry, i)
        elif isinstance(entry, ResubmitParsedEntry):
            region = parsed_regions_by_index[i]
        else:
            region = resolved_by_index[i]
        # Every entry kind echoes onto the part it rebuilds the posted operations its data
        # can run.
        operations = (
            entry.part.operations if isinstance(entry, ResubmitParsedEntry) else entry.operations
        )
        regions.append(with_runnable_operations(region, operations))

    # Resubmit closures create tasks on the event loop. Placeholders have none.
    regions = [
        _launch_trailing_tasks(
            region,
            launch_slot_options=params.stream_option_coverage,
            launch_observations=params.stream_observations,
        )
        for region in regions
    ]

    trailing_deadline = max(
        [params.deadline, *(r.reads_deadline for r in regions if r.reads_deadline is not None)]
    )
    degraded_leg = any(region.leg_degraded for region in regions)

    # Candidate regions carry the posted ordering override for the combiner.
    echo = QueryEcho(
        entry_path="dataset_resubmit",
        raw_query=inp.raw_query,
        normalized_query=None,
        extract_skipped=True,
    )

    if len(regions) == 1:
        region = regions[0]
        total_usage = region.llm_usage
        response = assemble_region(
            region,
            query="",
            extra_warnings=[],
            start_ms=start_ms,
            engine_build=build_commit(),
            max_candidates=params.max_candidates,
            include_observations=params.include_observations,
            include_skills=params.include_skills,
            skills_index=params.skills_index,
            llm_usage=total_usage,
            echo=echo,
        )
        if not params.stream_option_coverage and not params.stream_observations:
            for ev in response_sentence_events(response):
                yield ev
            for ev in _terminal_events(
                response, start_ms=start_ms, degraded_leg=degraded_leg, llm_usage=total_usage
            ):
                yield ev
            return
        for event in answer_events(response):
            yield event
        slot_options_warnings: list[Warning] = []
        late_departures: list[LateDepartures] = []
        async for ev in _stream_slot_options(
            [region],
            deadline=trailing_deadline,
            warnings_out=slot_options_warnings,
            late_departures_out=late_departures,
            # This door stores no response bundle, so its trailing losses need no report.
            incomplete_out=[],
            response=response,
            answered=[region],
        ):
            yield ev
        yield DoneEvent(
            elapsed_ms=now_ms() - start_ms,
            timed_out=False,
            degraded_leg=degraded_leg,
            trace_id=get_trace_id(),
            telemetry=DoneTelemetry(llm_usage=total_usage),
            warnings=slot_options_warnings,
            late_departures=late_departures,
        )
        return

    async with aclosing(
        _stream_combined_regions(
            regions,
            initial_llm_usage=None,
            query="",
            extra_warnings=[],
            echo=echo,
            timed_out=False,
            degraded_leg=degraded_leg,
            graph=graph,
            start_ms=start_ms,
            params=dataclasses.replace(params, deadline=trailing_deadline),
        )
    ) as events:
        async for event in events:
            yield event
    return
