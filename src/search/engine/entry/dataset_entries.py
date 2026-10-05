"""Resolve identity, candidates, and parsed entries for `dataset_resubmit`.

Candidate entries re-materialise each posted candidate through
`resolve_spec_resubmit` under the request semaphore. `_build_candidates_region`
sets `candidate_ordering_override` so the combiner preserves posted order instead
of re-sorting by `near_neighbor_fired`.

Parsed entries share the parsed door's interpretation (`parsed.interpret_parts`)
and per-part fan-out, so identical parts read the same dataset mentions, carry the
same warnings and count the same token usage on either door. Pins are confirmed per
part, while unpinned entity names share one resolution batch.
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
from search.engine.envelope import make_pipeline_step
from search.engine.errors import EngineInputError
from search.engine.fanout import _FanOutResult, _run_variables_streaming
from search.engine.identity_edit import synthesize_base
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region import RegionResult, resolve_spec_resubmit
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.region.resubmit_request import _requested_window_from_identity
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

_DIRECTIONAL_DECLARED_ROLES = ("creditor", "debtor")
"""Resolvable origin/destination roles that select the directional base branch."""


def spec_resubmit_input_for_identity(entry: ResubmitIdentityEntry) -> ResubmitIdentityPayload:
    """Copy an identity entry into its promote payload."""
    return ResubmitIdentityPayload(
        identity=entry.identity,
        base_spec_id=entry.base_spec_id,
        base_identity=entry.base_identity,
        removed=entry.removed,
        resolved_stat_var_dcids=entry.resolved_stat_var_dcids,
    )


def _candidate_promote_input(candidate: ResubmitCandidate) -> ResubmitIdentityPayload:
    """Build a promote payload with a base synthesized from the posted identity.

    The base preserves shape, entities, facets, and requested window, with empty
    slots because edit validation compares facets and entity roles. The posted
    identity's slots still undergo graph validation. Directional candidates pass
    `_DIRECTIONAL_DECLARED_ROLES` so one-sided origin or destination roles select
    the directional branch. Otherwise a destination-only candidate binds under
    `observationAbout` and fails `edit_incomplete`. Subject-only candidates use the subject branch.
    The placeholder role names do not affect the generated identity, which keys
    entities by axis.
    """
    identity = candidate.identity
    entities = identity.entities
    destination_dcids = entities.get("destination") or None
    origin_dcids = entities.get("origin") or None
    directional = bool(destination_dcids or origin_dcids)
    entity_dcids = destination_dcids if directional else entities.get("observationAbout") or None
    base_spec_id, base_identity = synthesize_base(
        identity.shape.model_dump(exclude_none=True),
        slots=[],
        entity_dcids=entity_dcids,
        origin_dcids=origin_dcids,
        resolved_stat_var_dcids=candidate.resolved_stat_var_dcids,
        requested_window=_requested_window_from_identity(identity.requested_window),
        facets=identity.facets,
        declared_roles=_DIRECTIONAL_DECLARED_ROLES if directional else None,
    )
    return ResubmitIdentityPayload(
        identity=identity,
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        removed=None,
        resolved_stat_var_dcids=candidate.resolved_stat_var_dcids,
    )


def _rewrite_candidate_pointer(
    exc: EngineInputError, *, index: int, candidate_index: int
) -> EngineInputError:
    """Point a refusal to the posted candidate fields or candidate slot.

    Re-map identity and resolved-dcid pointers directly. Point synthesized-field
    refusals at the candidate itself and preserve the original field in `detail`.
    Leave pointer-less refusals unchanged.
    """
    pointer = exc.pointer
    if pointer is None:
        return exc
    prefix = f"/input/entries/{index}/candidates/{candidate_index}"
    if pointer == "/input/identity" or pointer.startswith("/input/identity/"):
        new_pointer = prefix + pointer.removeprefix("/input")
        detail = exc.detail
    elif pointer == "/input/resolved_stat_var_dcids":
        new_pointer = prefix + "/resolved_stat_var_dcids"
        detail = exc.detail
    else:
        new_pointer = prefix
        original_detail = exc.detail or str(exc)
        detail = f"{original_detail} (refused field {pointer!r} was synthesized by the engine, not posted)"
    return EngineInputError(
        *exc.args,
        code=exc.code,
        status=exc.status,
        pointer=new_pointer,
        alternatives=exc.alternatives,
        detail=detail,
    )


def _merge_timing_by_step(regions: list[RegionResult]) -> dict[str, int]:
    """Sum step timings across candidate promotes for one part."""
    merged: dict[str, int] = {}
    for region in regions:
        for step, ms in region.timing_by_step.items():
            merged[step] = merged.get(step, 0) + ms
    return merged


def _candidates_collapsed_warning(*, posted: int, survivors: int) -> Warning:
    """Report a collapsed candidate set with posted and surviving counts.

    One survivor yields `candidates_collapsed_definite`. Zero yields
    `candidates_collapsed_no_data`.
    """
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
    """
    survivors = [r for r in promoted if r.status == "definite"]
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
        )
    if len(survivors) == 1:
        (only,) = survivors
        return dataclasses.replace(
            only,
            variable_text=entry.variable_text,
            warnings=(
                *only.warnings,
                _candidates_collapsed_warning(posted=len(promoted), survivors=1),
            ),
            timing_by_step=timing_by_step,
            earliest_index=index,
        )
    # Zero survivors. CandidateSet.specs has min_length=2, so there is no
    # "unsupported candidates" wire shape to fall back to either -- this
    # collapses to no_data regardless of which reason(s) the individual
    # promotes carried, preferring the first (posted-order)
    # promote's own reason when it has one, else the generic
    # reason resubmit_promote.py's survivor election uses for an edit whose
    # family holds no member.
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
            _candidates_collapsed_warning(posted=len(promoted), survivors=0),
        ),
        timing_by_step=timing_by_step,
        earliest_index=index,
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
            detail=message,
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
    sem: asyncio.Semaphore,
    coverage: CoverageSnapshot,
) -> RegionResult:
    """Promote candidates concurrently under the request semaphore.

    One semaphore bounds all promotes across up to 6 entries with 6 candidates
    each. Candidate edits skip observations and coverage probes. The caller checks
    the configured candidate cap before scheduling this coroutine.
    """

    async def _promote_one(candidate_index: int, candidate: ResubmitCandidate) -> RegionResult:
        async with sem:
            try:
                return await asyncio.to_thread(
                    resolve_spec_resubmit,
                    inp=_candidate_promote_input(candidate),
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
) -> tuple[dict[int, RegionResult], dict | None]:
    """Resolve parsed entries in one fan-out and return regions by posted index, with
    the token usage of the entries' one mention-reading call.

    Pins resolve per part. Unpinned names share one batch. Per-leg progress stays
    local because its indices are relative to this fan-out. This door emits no
    interpretation event. The fan-out cancels trailing tasks on infrastructure
    errors. On timeout, this function cancels partial-region tasks and raises
    `asyncio.TimeoutError`, preserving the fail-whole-request contract.
    """
    parts = [entry.part for _, entry in entries]

    # Confirm parsed pins per part before the mention reading, as the parsed door
    # does, then resolve unpinned names once and overlay each part's pins. Identity
    # and candidate entries resolve their own dcids.
    pins = await confirm_part_pins(
        parts,
        graph=graph,
        pointer_prefixes=[f"/input/entries/{entry_index}/part" for entry_index, _ in entries],
    )
    interpretation = await interpret_parts(parts, llm=llm, deadline=deadline)
    per_variable_pre_resolved = await per_part_pre_resolved(
        parts, pins, resolver=resolver, deadline=deadline
    )

    extract_step = make_pipeline_step("extract", ran=False)

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
        base_steps=[extract_step],
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
        per_variable_unit_texts=interpretation.unit_texts,
        per_variable_source_texts=interpretation.source_texts,
        per_variable_pre_resolved=per_variable_pre_resolved,
        per_variable_operations=interpretation.operations,
        per_variable_qualifiers=interpretation.qualifiers,
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
    return regions, interpretation.mention_usage
