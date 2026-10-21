"""Resolve identity, candidates, and parsed entries for `dataset_resubmit`.

Candidate entries re-materialise each posted candidate through
`resolve_spec_resubmit` under the request semaphore. `_build_candidates_region`
sets `candidate_ordering_override` so the part states the posted ordering.

Parsed entries share the parsed door's interpretation (`parsed.interpret_parts`)
and per-part fan-out, so identical parts carry the same inputs and warnings on either
door. Pins are confirmed per part, while unpinned entity names share one resolution
batch.
"""

from __future__ import annotations

import asyncio
import dataclasses

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from entity_resolution import Admission, EntityResolver
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.deadline import _cancel_trailing_tasks
from search.engine.entry.parsed import (
    confirm_part_pins,
    interpret_parts,
    per_part_pre_resolved,
)
from search.engine.errors import EngineInputError
from search.engine.fanout import _FanOutResult, _run_variables_streaming
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region import RegionResult, resolve_spec_resubmit
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.region.state import EngineOptions
from search.models import (
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
    ResubmitParsedEntry,
    Warning,
)
from series_coverage.shared import CoverageSnapshot


def spec_resubmit_input_for_identity(entry: ResubmitIdentityEntry) -> ResubmitIdentityPayload:
    """Copy an identity entry into its promote payload."""
    return ResubmitIdentityPayload(identity=entry.identity, entity_groups=entry.entity_groups)


# The promote payload's fields a posted candidate carries under the same names.
_CANDIDATE_FIELDS = ("/input/identity", "/input/entity_groups")


def _rewrite_candidate_pointer(
    exc: EngineInputError, *, index: int, candidate_index: int
) -> EngineInputError:
    """Point a refusal at the posted candidate's identity or entity groups, or at the candidate
    itself for a pointer outside them. Leave pointer-less refusals unchanged."""
    pointer = exc.pointer
    if pointer is None:
        return exc
    prefix = f"/input/entries/{index}/candidates/{candidate_index}"
    if any(pointer == field or pointer.startswith(f"{field}/") for field in _CANDIDATE_FIELDS):
        new_pointer = prefix + pointer.removeprefix("/input")
    else:
        new_pointer = prefix
    return EngineInputError(
        str(exc),
        code=exc.code,
        status=exc.status,
        pointer=new_pointer,
        alternatives=exc.alternatives,
    )


def _merge_timing_by_step(regions: list[RegionResult]) -> dict[str, int]:
    """Sum step timings across candidate promotes for one part."""
    merged: dict[str, int] = {}
    for region in regions:
        for step, ms in region.timing_by_step.items():
            merged[step] = merged.get(step, 0) + ms
    return merged


def _candidates_collapsed_warning(*, posted: int, resolved: int, survivors: int) -> Warning:
    """Report a collapsed candidate set with posted and surviving counts.

    One survivor yields `candidates_collapsed_definite`. Its message says the candidates
    rebuild to one reading when every posted candidate resolved, and says how many still
    resolve otherwise. Zero yields `candidates_collapsed_no_data`.
    """
    if survivors == 1 and resolved == posted:
        return Warning(
            code="candidates_collapsed_definite",
            severity="warn",
            message=(
                f"{posted} candidates were posted and all rebuild to 1 reading, so the "
                "part is reported as that answer and not as the posted candidate set."
            ),
        )
    if survivors == 1:
        return Warning(
            code="candidates_collapsed_definite",
            severity="warn",
            message=(
                f"{posted} candidates were posted; 1 still resolves to a definite "
                "answer, so the part is reported as that answer and not as the "
                "posted candidate set."
            ),
        )
    return Warning(
        code="candidates_collapsed_no_data",
        severity="warn",
        message=(
            f"{posted} candidates were posted; none resolves, so the part is "
            "reported as no_data and not as the posted candidate set."
        ),
    )


def _build_candidates_region(
    entry: ResubmitCandidatesEntry, promoted: list[RegionResult], *, index: int
) -> RegionResult:
    """Keep definite promotes in posted order, which `gather` preserves.
    No-data and unsupported promotes both drop. Two or more survivors rebuild
    the set. One becomes definite and zero becomes no-data, each with a warning.
    Each survivor carries its candidate's assumptions echo. A set reads each reading with
    its own, and a lone survivor's definite part states its own. Candidates that rebuild
    to one spec_id keep the first, with its echo, as a first-pass set keeps one reading
    per spec_id.
    """
    echoes = [
        None if candidate.assumptions is None else tuple(candidate.assumptions)
        for candidate in entry.candidates
    ]
    by_spec_id: dict[str, RegionResult] = {}
    for r, echo in zip(promoted, echoes, strict=True):
        if r.status == "definite":
            by_spec_id.setdefault(r.spec.spec_id, dataclasses.replace(r, assumption_echo=echo))
    survivors = list(by_spec_id.values())
    timing_by_step = _merge_timing_by_step(promoted)

    if len(survivors) >= 2:
        return RegionResult(
            variable_text=entry.variable_text,
            status="candidates",
            specs=tuple(r.spec for r in survivors),
            no_data_reason=None,
            warnings=tuple(w for r in survivors for w in r.warnings),
            departure_facts={
                key: fact for r in survivors for key, fact in r.departure_facts.items()
            },
            timing_by_step=timing_by_step,
            earliest_index=index,
            # Preserve posted ordering and truncation flags. A client-edited survivor
            # set cannot prove original truncation. The cap check only bounds its size.
            candidate_ordering_override=entry.ordering,
            candidate_budget_truncated=entry.truncated,
            # The first candidate's roles say which operations the set can run, as the
            # elected shape's do on a first pass.
            answered_roles=survivors[0].answered_roles,
            separate=entry.separate,
            reading_assumption_echoes={
                r.spec.spec_id: r.assumption_echo
                for r in survivors
                if r.assumption_echo is not None
            },
        )
    if len(survivors) == 1:
        (only,) = survivors
        return dataclasses.replace(
            only,
            variable_text=entry.variable_text,
            warnings=(
                *only.warnings,
                _candidates_collapsed_warning(
                    posted=len(promoted),
                    resolved=sum(r.status == "definite" for r in promoted),
                    survivors=1,
                ),
            ),
            timing_by_step=timing_by_step,
            earliest_index=index,
            separate=entry.separate,
        )
    # Zero survivors. CandidateSet.specs has min_length=2, so there is no
    # "unsupported candidates" wire shape to fall back to either -- this
    # collapses to no_data regardless of which reason(s) the individual
    # promotes carried, preferring the first (posted-order)
    # promote's own reason when it has one, else the generic
    # reason resubmit_promote.py's member election uses for a family that
    # holds no member.
    first = promoted[0]
    reason = first.no_data_reason if first.status == "no_data" else "no_matching_slice"
    return dataclasses.replace(
        first,
        variable_text=entry.variable_text,
        status="no_data",
        specs=(),
        no_data_reason=reason,
        unsupported=None,
        warnings=(
            *first.warnings,
            _candidates_collapsed_warning(posted=len(promoted), resolved=0, survivors=0),
        ),
        timing_by_step=timing_by_step,
        earliest_index=index,
        separate=entry.separate,
    )


def check_candidates_ceiling(entry: ResubmitCandidatesEntry, *, index: int) -> None:
    """Raise `request_invalid` when posted candidates exceed the deployment cap.

    The deployment advertises the cap via GET `/capabilities`. This config-only
    check runs before tasks, so static errors precede resolution failures.
    """
    if len(entry.candidates) > settings().max_candidates:
        message = (
            f"entry {index} posts {len(entry.candidates)} candidates, above this "
            f"deployment's ceiling of {settings().max_candidates}"
        )
        raise EngineInputError(
            message,
            code="request_invalid",
            pointer=f"/input/entries/{index}/candidates",
        )


async def resolve_candidates_entry(
    index: int,
    entry: ResubmitCandidatesEntry,
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    population_memo: OptionsPopulationMemo | None,
    place_options_pool: PlaceOptionsPool | None,
    deadline: float | None,
    full_read_deadline: float | None,
    sem: asyncio.Semaphore,
    coverage: CoverageSnapshot,
) -> RegionResult:
    """Promote candidates concurrently under the request semaphore, each member walk's full
    read bound by `full_read_deadline`.

    One semaphore bounds all promotes across up to 6 entries with 6 candidates
    each. Candidate edits skip observations and coverage probes. The caller checks
    the configured candidate cap before scheduling this coroutine.
    """

    async def _promote_one(candidate_index: int, candidate: ResubmitCandidate) -> RegionResult:
        async with sem:
            try:
                return await asyncio.to_thread(
                    resolve_spec_resubmit,
                    inp=ResubmitIdentityPayload(
                        identity=candidate.identity, entity_groups=candidate.entity_groups
                    ),
                    graph=graph,
                    sdmx=sdmx,
                    coverage=coverage,
                    options=EngineOptions(
                        materialise_graph=materialise_graph,
                        materialise_sdmx=materialise_sdmx,
                        rate_cache=rate_cache,
                        population_memo=population_memo,
                        place_options_pool=place_options_pool,
                        deadline=deadline,
                    ),
                    full_read_deadline=full_read_deadline,
                )
            except EngineInputError as exc:
                raise _rewrite_candidate_pointer(
                    exc, index=index, candidate_index=candidate_index
                ) from exc

    # Explicit tasks let `finally` cancel siblings. `gather` on bare coroutines
    # leaves them running after one promote fails.
    candidate_tasks = [
        asyncio.create_task(_promote_one(j, candidate))
        for j, candidate in enumerate(entry.candidates)
    ]
    try:
        promoted = await asyncio.gather(*candidate_tasks)
    finally:
        for t in candidate_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*candidate_tasks, return_exceptions=True)
    return _build_candidates_region(entry, list(promoted), index=index)


async def resolve_parsed_entries(
    entries: list[tuple[int, ResubmitParsedEntry]],
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    population_memo: OptionsPopulationMemo | None,
    place_options_pool: PlaceOptionsPool | None,
    pac: bool,
    deadline: float,
    include_option_coverage: bool,
    stream_slot_options: bool,
    include_observations: bool,
    stream_observations: bool,
    coverage: CoverageSnapshot,
) -> dict[int, RegionResult]:
    """Resolve parsed entries in one fan-out and return regions by posted index.

    Pins resolve per part. Unpinned names share one batch. Per-leg progress stays
    local because its indices are relative to this fan-out. This door emits no
    interpretation event. The fan-out cancels trailing tasks on infrastructure
    errors. On timeout, this function cancels partial-region tasks and raises
    `asyncio.TimeoutError`, preserving the fail-whole-request contract.
    """
    parts = [entry.part for _, entry in entries]

    # Confirm pins per part, then resolve unpinned names once and overlay each part's
    # pins. Identity and candidate entries resolve their own dcids.
    pointer_prefixes = [f"/input/entries/{entry_index}/part" for entry_index, _ in entries]
    pins = await confirm_part_pins(
        parts, graph=graph, resolver=resolver, pointer_prefixes=pointer_prefixes
    )
    interpretation = interpret_parts(parts, pointer_prefixes=pointer_prefixes)
    per_variable_pre_resolved = await per_part_pre_resolved(
        parts, pins, resolver=resolver, deadline=deadline
    )

    out = _FanOutResult()
    async for _ev in _run_variables_streaming(
        interpretation.variable_texts,
        entities=[],
        date_request=None,
        query="",
        pac=pac,
        graph=graph,
        resolver=resolver,
        admission=admission,
        llm=llm,
        sdmx=sdmx,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        base_timing={},
        pre_resolved=None,
        deadline=deadline,
        degrade_on_error=True,
        use_semaphore=True,
        out=out,
        include_option_coverage=include_option_coverage,
        stream_slot_options=stream_slot_options,
        include_observations=include_observations,
        stream_observations=stream_observations,
        per_variable_entities=interpretation.entities,
        per_variable_date_requests=interpretation.date_requests,
        per_variable_role_queries=interpretation.role_queries,
        per_variable_entity_roles=interpretation.entity_roles,
        per_variable_member_entities=interpretation.member_entities,
        per_variable_unit_texts=interpretation.unit_texts,
        per_variable_source_scopes=interpretation.source_scopes,
        per_variable_source_codes=interpretation.source_codes,
        per_variable_pre_resolved=per_variable_pre_resolved,
        per_variable_operations=interpretation.operations,
        per_variable_open_axes=interpretation.open_axes,
        per_variable_separate=interpretation.separate,
        coverage=coverage,
    ):
        # Drain local-index progress events. The fan-out cleans up its regions
        # before propagating infrastructure errors or cancellation.
        pass

    if out.timed_out:
        # Fan-out timeout leaves partial regions and live tasks. Cancel them before failing.
        await _cancel_trailing_tasks(out.regions)
        raise TimeoutError("dataset_resubmit: a parsed entry exceeded the pipeline deadline")

    regions: dict[int, RegionResult] = {}
    for j, (index, _entry) in enumerate(entries):
        regions[index] = dataclasses.replace(
            interpretation.interpreted(out.regions[j], j), earliest_index=index
        )
    return regions
