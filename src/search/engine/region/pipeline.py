"""Orchestrate per-variable search through recall, binding, materialisation, and answer assembly."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Final, Literal, cast

import dc_graph
from dc_sdmx import SdmxCapExceededError
from entity_resolution import Admission, EntityResolver, Resolution
from materialise.member_conversion import request_conversion_reader
from origin_catalogue import recording_un_blocked_places
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.envelope import _sum_usage, now_ms
from search.engine.errors import (
    DeclaredDefaultUnsupportedError,
    DefaultScopeIndexSourcedError,
    EngineInfraError,
)
from search.engine.events import Event, EventSink, SentenceEvent
from search.engine.extract import DateRequest
from search.engine.interpret import Recall
from search.engine.qualifier_match import CodedQualifier
from search.engine.region.candidate_admission import alternates_unevaluated_warning
from search.engine.region.candidates import resolve_candidates
from search.engine.region.codes import ALTERNATES_UNEVALUATED, UN_SYSTEM_UNAVAILABLE
from search.engine.region.near_neighbor import _representative_score_is_weak
from search.engine.region.phase_answer import _phase_assemble_definite, _phase_ground_answer
from search.engine.region.phase_election import _phase_repair_bindings, _phase_vet_election
from search.engine.region.phase_intake import (
    _phase_check_entities,
    _phase_derive_shapes,
    _phase_recall,
    _phase_resolve_places,
)
from search.engine.region.phase_place_materialise import (
    _phase_facet_requests,
    _phase_materialise,
    _phase_no_data_answer,
    _phase_place_bindings,
)
from search.engine.region.phase_rank_bind import (
    _admitted_qualifier_phrases,
    _phase_bind,
    _phase_prune_and_rank,
)
from search.engine.region.place_roles import group_request_names
from search.engine.region.related_read import cancel_related_reads, settle_related_reads
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions, ResolveState
from search.engine.region.un_system import with_un_unavailable_warning
from search.engine.region.withheld import with_withheld_warning
from search.engine.retrieval.bound_values import RoleClassWalks
from search.engine.role_text import DoorRole
from search.models import Operation, RoleAxis, Warning
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

# No-data reasons about the request's entities. The second pass fills a gap in our variables,
# so it does not run after these.
_ENTITY_TERMINALS: Final = frozenset(
    {"entity_not_resolved", "entity_ambiguous", "entity_role_not_admitted"}
)

# The second pass ends this long before the request deadline. The fan-out cancels a leg still
# running at the deadline and serves a placeholder in place of its answer, so the leg needs
# this time to unwind the cancelled pass, run the `withheld_by_registry` re-read (one facet
# read, which the graph's periodic stall can hold for 2 to 4 s) and queue the first
# pass's answer.
_SECOND_PASS_RESERVE_S: Final = 3.0

# Second-pass failures that mean an upstream read failed or hit a capacity limit. Each leaves
# the first answer standing with `un_system_unavailable`. Any other exception is a defect,
# logged with its traceback, and leaves the first answer standing unwarned.
_UPSTREAM_FAILURES: Final = (
    EngineInfraError,
    TimeoutError,
    dc_graph.GraphReadError,
    dc_graph.GraphCapExceededError,
    SdmxCapExceededError,
)

_SECOND_PASS_INCOMPLETE: Final = (
    "The search over the UN System Data Commons' variables did not complete; "
    "candidates come from our Data Commons only."
)


async def resolve_variable(
    variable: str,
    *,
    entities: list[str],
    date_request: DateRequest | None,
    detect_query: str,
    role_query: str,
    entity_roles: dict[str, list[DoorRole]] | None = None,
    dropped_qualifier_phrases: list[str] | None = None,
    operations: Sequence[Operation] = (),
    open_axes: frozenset[RoleAxis] = frozenset(),
    query_text: str | None = None,
    qualifiers: tuple[CodedQualifier, ...] | None = None,
    pac: bool,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission | None = None,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    base_timing: dict[str, int],
    pre_resolved: dict[str, Resolution] | None = None,
    variable_index: int = 0,
    emit: EventSink | None = None,
    unit_text: str | None = None,
    unit_text_best_effort: bool = False,
    source_text: str | None = None,
    named_dataset: str | None = None,
    coverage: CoverageSnapshot,
    options: EngineOptions | None = None,
) -> RegionResult:
    """Run recall, binding, materialisation, and answer assembly for one variable.

    `detect_query` supplies recall text, including typed qualifier phrases. The bare `variable`
    feeds binding, qualifier matching, and deduplication. `role_query` checks door-supplied axes
    against query wording and never supplies one. Raw-text doors share the full query across legs. Parsed doors use a
    per-part query. `entity_roles` maps surface names to door-supplied flow axes. An empty map
    treats entities as undirected.

    `dropped_qualifier_phrases` are the request's match text, plus on the raw-text door this
    variable's own operation phrases, which bind treats alike. `operations` are the leg's
    admitted operations; one over `category` opens the slots its phrase names on the
    elected shape whether or not bind claims them. `open_axes` are the flow axes the leg
    asks every member of (`ResolveState.open_axes`). `query_text` adds
    the verbatim query as a match source for a single-variable request. `qualifiers` adds a
    typed source. Its phrase-relative spans stay separate from variable spans. The raw-text door
    supplies both qualifier inputs.

    `admission` checks derived shape roles before election. Direct callers default to
    `Admission(graph=graph)`. The request path supplies a shared instance. `resolver` resolves
    named entities during recall and place expansion. `pre_resolved` avoids per-variable
    resolver calls for shared entities. `pac` enables place-as-constraint. `coverage` is the
    required request snapshot, shared with materialisation, unit decisions, and observation
    dispatch.

    `emit` adds `shape` and `sentence` events concurrently with pipeline work. `variable_index`
    correlates them. `named_dataset` is the declared dataset a parsed part names in
    `ParsedPart.provenance`, which shape candidacy reads (`ResolveState.named_dataset`).
    `unit_text` and `source_text` carry parsed-door requests, and
    `unit_text` the raw-text door's typed unit phrase. A bound, expressible unit narrows
    observation facets. With `unit_text_best_effort`, a unit the elected data cannot convert
    to is withdrawn and the leg answers unconverted (`phase_shared.withdraw_typed_unit`).
    `options` configures enrichment, streaming, clients, caches, and the deadline. `None` uses default `EngineOptions()` settings.

    `graph` and `sdmx` are shared, thread-safe clients. Dev-finance role-keyed observation and
    availability reads use `sdmx`. `llm` is stateless per call. `date_request` is the shared
    window, while `base_timing` carries extraction timing.

    Recall runs once. The pipeline answers from our candidates first; when that pass ends
    `no_data`, it waits for the UN System Data Commons resolve that recall started and runs
    again on its candidates (`_fill_gap`). A part answered on the first pass never waits for
    that resolve.

    Returns a `RegionResult` with status `definite`, `candidates`, `no_data`, or `unsupported`.
    When the pipeline's UN System Data Commons reads named places the UN front door refuses,
    the result carries `un_system_unavailable` naming them. The request's first subject-only
    no-data answer with an elected shape carries `withheld_by_registry` when the catalogue
    withholds the provenances holding its data (`region.withheld`). The result is `degraded`
    when the call swallowed a read (`degraded_reads`).
    """
    _admission = admission if admission is not None else Admission(graph=graph)
    _options = options if options is not None else EngineOptions()

    # One conversion memo serves every unit decision, coverage check and preview of the call.
    materialise_graph = (
        request_conversion_reader(_options.materialise_graph)
        if _options.materialise_graph is not None
        else None
    )
    role_class_walks = (
        RoleClassWalks(_options.materialise_graph)
        if _options.materialise_graph is not None
        else None
    )

    def new_state(
        *,
        pool: Literal["ours", "un_system"],
        emit: EventSink | None,
        deadline: float | None,
        timing: dict[str, int],
    ) -> ResolveState:
        return ResolveState(
            variable=variable,
            entities=entities,
            date_request=date_request,
            detect_query=detect_query,
            role_query=role_query,
            entity_roles=entity_roles or {},
            dropped_qualifier_phrases=dropped_qualifier_phrases or [],
            operations=tuple(operations),
            open_axes=open_axes,
            query_text=query_text,
            qualifiers=qualifiers,
            pac=pac,
            graph=graph,
            resolver=resolver,
            admission=_admission,
            llm=llm,
            sdmx=sdmx,
            deadline=deadline,
            pre_resolved=pre_resolved,
            variable_index=variable_index,
            emit=emit,
            include_option_coverage=_options.include_option_coverage,
            stream_slot_options=_options.stream_slot_options,
            include_observations=_options.include_observations,
            stream_observations=_options.stream_observations,
            unit_text=unit_text,
            unit_text_best_effort=unit_text_best_effort,
            source_text=source_text,
            named_dataset=named_dataset,
            materialise_graph=materialise_graph,
            materialise_sdmx=_options.materialise_sdmx,
            rate_cache=_options.rate_cache,
            population_memo=_options.population_memo,
            place_options_pool=_options.place_options_pool,
            role_class_walks=role_class_walks,
            coverage=coverage,
            pool=pool,
            timing=timing,
            warnings=[],
            var_usage=None,
            # A `no_matching_slice` deferral needs the place-less mint to attach
            # `flagged_match`, so it sets this to keep the single-shape path.
            force_single_shape_path=False,
            # A bound, expressible unit narrows observation facets and enters the
            # spec identity, fetch facet filter, and resolved unit request basis.
            bound_unit_outcome=None,
            unit_facet_filter=None,
        )

    # Over a graph that can declare a default facet, the first pass holds its sentence events
    # until the part's answer stands (`_fill_gap`), so none reaches the stream ahead of a
    # declared-default refusal a later step raises: the elected shape's materialise, a
    # sibling's re-election, a candidate fan-out or the UN second pass. A graph declaring
    # none cannot refuse, so its sentences stream as they are built.
    first_sentences: list[SentenceEvent] = []
    state = new_state(
        pool="ours",
        emit=(
            _holding_sentences(emit, first_sentences)
            if emit is not None and graph.declares_facet_defaults()
            else emit
        ),
        deadline=_options.deadline,
        timing=dict(base_timing),
    )
    # A read the pass swallows marks the part degraded, so no cache stores it.
    with recording_degraded_reads() as lost:
        with recording_un_blocked_places() as un_blocked:
            await _phase_recall(state)
            rcl = cast(Recall, state.rcl)
            # `_phase_resolve_places` rewrites both entity maps in place, so the second pass
            # starts from recall's own.
            recalled = rcl.model_copy(
                update={
                    "resolutions": dict(rcl.resolutions),
                    "resolved_entity_names": dict(rcl.resolved_entity_names),
                }
            )
            result = await _run_phases(state)
            result = await _fill_gap(
                result,
                rcl=recalled,
                deadline=_options.deadline,
                variable_index=variable_index,
                emit=emit,
                first_sentences=first_sentences,
                first_measure_class_refused=state.n_measure_class_refused,
                first_names_dataset=state.named_dataset is not None
                or bool(state.excluded_datasets),
                second_state=lambda sink, bound: new_state(
                    pool="un_system",
                    emit=sink,
                    deadline=bound,
                    timing={},
                ),
            )
        result = with_un_unavailable_warning(result, un_blocked)
        result = await with_withheld_warning(result, state.graph, deadline=state.deadline)
    return dataclasses.replace(result, degraded=True) if lost.degraded else result


def _holding_sentences(emit: EventSink, held: list[SentenceEvent]) -> EventSink:
    """`emit` with each sentence event appended to `held` instead of sent."""

    async def sink(event: Event) -> None:
        if isinstance(event, SentenceEvent):
            held.append(event)
        else:
            await emit(event)

    return sink


async def _fill_gap(
    first: RegionResult,
    *,
    rcl: Recall,
    deadline: float | None,
    variable_index: int,
    emit: EventSink | None,
    first_sentences: list[SentenceEvent],
    first_measure_class_refused: int,
    first_names_dataset: bool,
    second_state: Callable[[EventSink | None, float | None], ResolveState],
) -> RegionResult:
    """`first`, or the answer from a second pass over `rcl.un_candidates` when `first` is
    `no_data` and its part names or excludes no dataset.

    A part that names a dataset, by the parsed door's `provenance`, its source slot or
    electing an opt-in table's own measure, or that excludes one through its source slot
    (`first_names_dataset`, read from `ResolveState.named_dataset` and
    `ResolveState.excluded_datasets` after vetting), answers from the declared datasets it
    chose, so its `first` stands: the UN System Data Commons publishes none of them, and
    an answer from it would serve what the part named another table for or excluded.
    The second pass shares recall and the resolved entities, and ends `_SECOND_PASS_RESERVE_S`
    before `deadline`. It first waits for recall's UN resolve, which is bounded by
    `SEARCH_UN_RESOLVE_TIMEOUT_S`, and not past that bound. Its answer stands only when it is
    `definite` or `candidates`; otherwise `first` stands with its reason. A `first` holding an
    `alternate_match`, a reading the model nominated from our candidates with data at the
    place, also stands when the second pass's elected shape matches the phrase only weakly
    (`_representative_score_is_weak`): such an answer has no better claim to the question than
    the alternate. The result carries both passes' step timings and model usage, and states
    `alternates_unevaluated` when either pass left an alternate unevaluated: the first pass's
    `no_data` may rest on the alternate it left unread, and a first pass that stands may have
    been filled by a completed read. `first` stands whenever the second pass fails, except on
    a declared-default refusal (`DeclaredDefaultUnsupportedError`) or an index-sourced shape
    meeting a default scope (`DefaultScopeIndexSourcedError`), which propagate. A failed UN
    resolve, or a second pass that runs out of time, fails upstream
    (`_UPSTREAM_FAILURES`) or has no time to start, warns `un_system_unavailable`, only ever on
    a `no_data` answer, the one case where the failure could have changed it. Cancellation
    propagates.

    Shape and step events on `emit` describe the first pass. The second pass holds its
    sentence events, and so does the first over a graph that declares default facets
    (`first_sentences`, else empty, its sentences already sent). Only the pass whose answer
    stands forwards them, once nothing can refuse, so the stream carries no sentence ahead of
    a refusal and the last sentence on it matches the answer.
    """

    async def stands(result: RegionResult, sentences: list[SentenceEvent]) -> RegionResult:
        if emit is not None:
            for event in sentences:
                await emit(event)
        return result

    bound = None if deadline is None else deadline - _SECOND_PASS_RESERVE_S
    skip = await _second_pass_skip(first, rcl=rcl, bound=bound, names_dataset=first_names_dataset)
    if skip is not None:
        _log_gap_fill(
            variable_index,
            first,
            rcl,
            pass1_measure_refused=first_measure_class_refused,
            pass2=f"skipped:{skip}",
        )
        if skip == "un_unavailable":
            return await stands(
                _with_un_search_warning(
                    first,
                    "The UN System Data Commons did not answer the variable search; "
                    "candidates come from our Data Commons only.",
                ),
                first_sentences,
            )
        if skip == "deadline":
            return await stands(
                _with_un_search_warning(first, _SECOND_PASS_INCOMPLETE), first_sentences
            )
        return await stands(first, first_sentences)

    sentences: list[SentenceEvent] = []

    async def hold_sentence(event: Event) -> None:
        if isinstance(event, SentenceEvent):
            sentences.append(event)

    state = second_state(hold_sentence if emit is not None else None, bound)
    state.rcl = rcl
    t0 = now_ms()
    try:
        async with asyncio.timeout(None if bound is None else bound - time.monotonic()):
            second = await _run_phases(state)
    except Exception as exc:
        _log_gap_fill(
            variable_index,
            first,
            rcl,
            pass1_measure_refused=first_measure_class_refused,
            pass2="ran",
            pass2_status=f"error:{type(exc).__name__}",
            pass2_ms=now_ms() - t0,
            pass2_below_floor=state.n_below_score_floor,
            pass2_measure_refused=state.n_measure_class_refused,
        )
        if isinstance(exc, (DeclaredDefaultUnsupportedError, DefaultScopeIndexSourcedError)):
            raise
        if isinstance(exc, _UPSTREAM_FAILURES):
            return await stands(
                _with_un_search_warning(first, _SECOND_PASS_INCOMPLETE), first_sentences
            )
        logger.exception("gap_fill part=%d: the second pass failed", variable_index)
        return await stands(first, first_sentences)
    elected = state.shape_draft
    _log_gap_fill(
        variable_index,
        first,
        rcl,
        pass1_measure_refused=first_measure_class_refused,
        pass2="ran",
        pass2_status=f"{second.status}/{second.no_data_reason}",
        pass2_ms=now_ms() - t0,
        pass2_below_floor=state.n_below_score_floor,
        pass2_measure_refused=state.n_measure_class_refused,
        pass2_score=None if elected is None else elected.measured_score,
    )

    displaces_alternate = first.alternate_match is None or not _representative_score_is_weak(
        None if elected is None else elected.measured_score
    )
    chosen = (
        second if second.status in ("definite", "candidates") and displaces_alternate else first
    )
    await stands(chosen, sentences if chosen is second else first_sentences)
    timing = dict(first.timing_by_step)
    for step, ms in second.timing_by_step.items():
        timing[step] = timing.get(step, 0) + ms
    other = first if chosen is second else second
    carried = (
        ()
        if chosen.alternates_left_out
        else tuple(w for w in other.warnings if w.code == ALTERNATES_UNEVALUATED)
    )
    return dataclasses.replace(
        chosen,
        warnings=(*chosen.warnings, *carried),
        timing_by_step=timing,
        llm_usage=_sum_usage(first.llm_usage, second.llm_usage),
        scope_unsettled=first.scope_unsettled or second.scope_unsettled,
    )


async def _second_pass_skip(
    first: RegionResult, *, rcl: Recall, bound: float | None, names_dataset: bool
) -> str | None:
    """Why the second pass does not run after `first`, or `None` when it runs. `bound` is
    when the second pass must end. `names_dataset` holds when the part names or excludes a
    dataset. Only a `no_data` answer that the second pass could fill waits for the UN
    resolve, and not past `bound`."""
    if first.status != "no_data":
        return "answered"
    if first.no_data_reason in _ENTITY_TERMINALS:
        return "entity"
    if names_dataset:
        return "named_dataset"
    try:
        async with asyncio.timeout(None if bound is None else bound - time.monotonic()):
            await rcl.settle_un()
    except TimeoutError:
        return "deadline"
    if rcl.un_unavailable:
        return "un_unavailable"
    if not rcl.un_candidates:
        return "no_un_candidates"
    if bound is not None and time.monotonic() >= bound:
        return "deadline"
    return None


def _stating_unevaluated(result: RegionResult, state: ResolveState) -> RegionResult:
    """`result`, stating `alternates_unevaluated` when the pass left an alternate
    unevaluated, on every path that reached admission, a capped alternate's refusal
    included. The warning alone marks the result as leaving an alternate out
    (`RegionResult.alternates_left_out`), which keeps it out of the response cache."""
    if not state.unevaluated_alternates:
        return result
    assert state.alternates_presence_failure is not None
    warning = alternates_unevaluated_warning(state.alternates_presence_failure)
    return dataclasses.replace(result, warnings=(*result.warnings, warning))


def _with_un_search_warning(result: RegionResult, message: str) -> RegionResult:
    warning = Warning(code=UN_SYSTEM_UNAVAILABLE, severity="warn", message=message)
    return dataclasses.replace(result, warnings=(*result.warnings, warning))


def _log_gap_fill(
    variable_index: int,
    first: RegionResult,
    rcl: Recall,
    *,
    pass1_measure_refused: int,
    pass2: str,
    pass2_status: str = "-",
    pass2_ms: int = 0,
    pass2_below_floor: int = 0,
    pass2_measure_refused: int = 0,
    pass2_score: float | None = None,
) -> None:
    """One line per part for the gap-fill measurement. `un_candidates` reads `pending` while
    the UN resolve is still running; `pass2_below_floor` counts the UN shapes the
    `SEARCH_UNKNOWN_SCORE_THRESHOLD` floor dropped from the second pass's menu, each pass's
    `measure_refused` counts the shapes the measure-class check dropped from its menu
    (`ResolveState.n_measure_class_refused`), and `pass2_score` is the second pass's elected
    shape's recall score (`ShapeDraft.measured_score`)."""
    un_candidates = len(rcl.un_candidates) if rcl.un_leg.done() else "pending"
    logger.info(
        "gap_fill part=%d pass1=%s/%s pass1_measure_refused=%d un_candidates=%s pass2=%s "
        "pass2_status=%s pass2_ms=%d pass2_below_floor=%d pass2_measure_refused=%d "
        "pass2_score=%s",
        variable_index,
        first.status,
        first.no_data_reason,
        pass1_measure_refused,
        un_candidates,
        pass2,
        pass2_status,
        pass2_ms,
        pass2_below_floor,
        pass2_measure_refused,
        "-" if pass2_score is None else f"{pass2_score:.4f}",
    )


async def _run_phases(state: ResolveState) -> RegionResult:
    """Run `_PHASES` in order until one returns the terminal result. However the pass ends,
    it then cancels what it queued (`_cancel_queued_reads`) and ends its related-measures
    reads (`_end_related_reads`)."""
    try:
        for phase in _PHASES:
            # resolve_candidates alone takes rate_cache; other phase callables take only state.
            if phase is resolve_candidates:
                result = await phase(state, rate_cache=state.rate_cache)
            else:
                result = await phase(state)
            if result is not None:
                # Only a vetted election can report qualifier matches: earlier terminals may
                # hold pre-bind candidates without a forced binding or elected shape.
                admitted = (
                    _admitted_qualifier_phrases(state.admitted_qualifiers)
                    | state.satisfied_measure_phrases
                    | state.elected_tie_phrases
                    | state.applied_dropped_qualifier_phrases
                    | state.unapplied_qualifier_phrases
                    | state.slot_default_phrases
                    if state.shape_draft is not None
                    else frozenset()
                )
                return _stating_unevaluated(
                    dataclasses.replace(
                        result,
                        admitted_qualifier_phrases=admitted,
                        scope_unsettled=_scope_unsettled(state),
                        entity_words=group_request_names(state),
                    ),
                    state,
                )
        raise AssertionError("_phase_assemble_definite is terminal")
    finally:
        _cancel_queued_reads(state)
        await _end_related_reads(state)


def _scope_unsettled(state: ResolveState) -> bool:
    """Whether the pass's default-scope presence read left a StatVar's scope unsettled
    (`RegionResult.scope_unsettled`)."""
    return state.default_presence is not None and bool(state.default_presence.unsettled)


def _cancel_queued_reads(state: ResolveState) -> None:
    """Cancel the pass's member walks (`state.alternate_walks`) and its untaken listing of
    Earth's countries (`state.countries_listing`). A walk or listing still queued is
    cancelled, and a running one finishes with its result dropped."""
    for walk in state.alternate_walks.values():
        walk.cancel()
    if state.countries_listing is not None:
        state.countries_listing.cancel()


async def _end_related_reads(state: ResolveState) -> None:
    """Wait for every related-measures read the pass started (`state.related_reads`), so
    none outlives the answer. A pass being cancelled cancels them instead, since waiting
    would hold its cancellation."""
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        cancel_related_reads(state.related_reads)
    else:
        await settle_related_reads(state.related_reads)


# `_PHASES` follows `NO_DATA_RESOLUTION_ORDER`. Each phase mints reasons ranked after those
# available earlier, so the first terminal is the earliest applicable reason.
# Entity errors come from `_phase_check_entities` and `_unresolved_entity_terminal` before
# earlier shape or bind exits. `entity_ambiguous` means every unresolved entity has an
# Ambiguous resolution.
# `_phase_derive_shapes` mints `entity_role_not_admitted` when each shape fails to admit at
# least one resolved request entity. It mints `role_signature_incompatible` for
# mismatched signatures across candidate groups.
# `variable_not_resolved` comes from derive_shapes (no confirmed shape), prune/rank (no eligible
# shape), bind (ask or out-of-range election), or repair (degenerate binding).
# `_phase_place_bindings` mints `role_not_resolved` for unresolved multi-place axes, a solitary
# place without a supplied axis, or contradicted wording. It mints `role_incompatible` for a
# directional query against a non-role shape or an unplaceable role entity.
# Materialise can also mint `role_not_resolved` when declared roles miss destination vocabulary.
# `_phase_facet_requests` handles `unit_not_resolved` when a bound unit fails parsing. It also
# rejects inexpressible units and bare data-source requests as unsupported.
# `qualifier_not_bound` comes from definite assembly after identity confirmation or candidates'
# single-survivor collapse. Both paths call `_unbound_query_qualifiers`. Prune/rank mints it
# for a named dataset that publishes no recalled shape, and vetting for a source slot that
# excludes every dataset the part may answer from.
# Materialise can mint `no_matching_slice`, deferred to the place-less mint, or `binding_mismatch`.
# No-data/candidate handlers forward materialised `binding_mismatch` reasons.
# `_phase_materialise` reports `denominator_not_available` after `no_matching_slice` is resolved.
# `no_observations` comes from no-data handling, candidates, grounding's window answer, or
# definite assembly.
# `facet_not_available` comes from candidates/assembly for an unmatched unit facet, or grounding
# before fetch when no convertible facet can be selected.
# `resolve_variable` runs `_phase_recall` once, before the first pass.
_PHASES: tuple[Callable[[ResolveState], Awaitable[RegionResult | None]], ...] = (
    _phase_resolve_places,
    _phase_derive_shapes,
    _phase_prune_and_rank,
    _phase_bind,
    _phase_vet_election,
    _phase_repair_bindings,
    _phase_check_entities,
    _phase_place_bindings,
    _phase_facet_requests,
    _phase_materialise,
    resolve_candidates,
    _phase_no_data_answer,
    _phase_ground_answer,
    _phase_assemble_definite,
)
