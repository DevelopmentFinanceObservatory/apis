"""Orchestrate per-variable search through recall, binding, materialisation, and answer assembly."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import ExitStack
from typing import Final

import dc_graph
from code_index.types import PartyWord
from dc_sdmx import SdmxCapExceededError
from entity_resolution import Admission, EntityResolver, Resolution
from materialise.member_conversion import request_conversion_reader
from origin_catalogue import note_un_unserved, recording_un_blocked_places, scoped_reads
from search.engine.binders import Binder, GeminiBinder
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM, recording_llm_usage
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.conjoin import NESTED_SHAPE_STEPS
from search.engine.deadline import _cancel_trailing_tasks
from search.engine.degraded_reads import (
    DegradedReads,
    note_degraded_read,
    recording_degraded_reads,
)
from search.engine.envelope import _sum_usage, now_ms
from search.engine.errors import (
    RATE_TABLE_REFUSALS,
    DeclaredDefaultUnsupportedError,
    DefaultScopeIndexSourcedError,
    EngineInfraError,
)
from search.engine.events import Event, EventSink
from search.engine.extract import DateRequest
from search.engine.leg_phase import enter_phase
from search.engine.qualifier_match import CodedQualifier
from search.engine.region.candidate_admission import alternates_unevaluated_warning
from search.engine.region.codes import ALTERNATES_UNEVALUATED, UN_SYSTEM_UNAVAILABLE
from search.engine.region.decisions import PartyReadingWithdrawn
from search.engine.region.phase_answer import _phase_outcome
from search.engine.region.phase_election import _phase_settle
from search.engine.region.phase_intake import (
    _phase_check_entities,
    _phase_derive_shapes,
    _phase_recall,
    _phase_resolve_places,
    un_pool_shapes,
)
from search.engine.region.phase_place_materialise import (
    _phase_facet_requests,
    _phase_materialise,
    _phase_place_bindings,
)
from search.engine.region.phase_rank_bind import (
    _admitted_qualifier_phrases,
    _phase_bind,
    _phase_prune_and_rank,
    exclusion_records,
)
from search.engine.region.phase_shared import settle_unjudged_refusal
from search.engine.region.place_roles import group_request_names, stated_party_words
from search.engine.region.related_read import cancel_related_reads, settle_related_reads
from search.engine.region.requirements import unbinding_phrases, with_requirement_departures
from search.engine.region.result import RegionResult
from search.engine.region.source_choice import (
    allows_un_system,
    answer_provenances,
    idle_inclusion_warnings,
    with_idle_inclusions,
)
from search.engine.region.state import EngineOptions, ResolveState
from search.engine.region.un_system import with_un_unavailable_warning
from search.engine.region.weak_score import _representative_score_is_weak
from search.engine.region.withheld import elected_series, with_withheld_warning
from search.engine.retrieval.bound_values import RoleClassWalks, elects_within, held_family_memo
from search.engine.role_text import DoorRole
from search.engine.shape import Pool
from search.engine.source_code import SeriesCodeMembers
from search.engine.source_scope import SourceScope
from search.models import EntryPath, Operation, RoleAxis, Warning
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

# No-data reasons about the request's entities. The UN pass fills a gap in our variables,
# so it does not run after these.
_ENTITY_TERMINALS: Final = frozenset(
    {"entity_not_resolved", "entity_ambiguous", "entity_set_too_large", "entity_role_not_admitted"}
)

# The UN pass ends this long before the request deadline. The fan-out cancels a leg still
# running at the deadline and serves a placeholder in place of its answer, so the leg needs
# this time to unwind the cancelled pass, run the `withheld_by_registry` re-read (one facet
# read, which the graph's periodic stall can hold for 2 to 4 s) and queue our pass's
# answer.
_UN_PASS_RESERVE_S: Final = 3.0

# Failures that mean an upstream read failed or hit a capacity limit. In the UN pass each
# leaves our pass's answer standing with `un_system_unavailable`; any other exception there
# is a defect, logged with its traceback, and leaves our answer standing unwarned. In a
# party reading each is a failed read, noted as degraded when the reading is withdrawn
# (`resolve_variable`).
_UPSTREAM_FAILURES: Final = (
    EngineInfraError,
    TimeoutError,
    dc_graph.GraphReadError,
    dc_graph.GraphCapExceededError,
    SdmxCapExceededError,
)

# Refusals of the whole request that a pass raises on purpose (`_fill_gap`).
_REQUEST_REFUSALS: Final = (
    DeclaredDefaultUnsupportedError,
    DefaultScopeIndexSourcedError,
    *RATE_TABLE_REFUSALS,
)

_UN_PASS_INCOMPLETE: Final = (
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
    member_entities: frozenset[str] = frozenset(),
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
    binder: Binder | None = None,
    sdmx: EngineSdmxClient,
    base_timing: dict[str, int],
    pre_resolved: dict[str, Resolution] | None = None,
    entity_surfaces: dict[str, str] | None = None,
    variable_index: int = 0,
    emit: EventSink | None = None,
    unit_text: str | None = None,
    unit_text_best_effort: bool = False,
    source_scope: SourceScope | None = None,
    source_code: SeriesCodeMembers | None = None,
    coverage: CoverageSnapshot,
    options: EngineOptions | None = None,
    door: EntryPath = "parsed",
) -> RegionResult:
    """Run recall, binding, materialisation, and answer assembly for one variable.

    `detect_query` supplies recall text, including typed qualifier phrases. The bare `variable`
    feeds binding, qualifier matching, and deduplication. `role_query` checks door-supplied axes
    against query wording and never supplies one. Raw-text doors share the full query across legs. Parsed doors use a
    per-part query. `entity_roles` maps surface names to door-supplied flow axes. An empty map
    treats entities as undirected. `member_entities` are the entities the door asks for as
    their members (`ResolveState.member_entities`).

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
    resolver calls for shared entities. `entity_surfaces` maps an entity name to the query
    words recall resolves it through (`surface_resolution.resolve_names`). `pac` enables place-as-constraint. `coverage` is the
    required request snapshot, shared with materialisation, unit decisions, and observation
    dispatch.

    `emit` adds `frame`, `shape` and `step` events concurrently with pipeline work.
    `variable_index` correlates them. The part's one `sentence` event goes out with the
    response (`sentence_events.response_sentence_events`). `source_scope` is the provenances the part's source mentions allow
    (`search.engine.source_scope`). Shape candidacy and election read it
    (`ResolveState.scope`), both passes read through the request's catalogue scoped to it
    (`origin_catalogue.Catalogue.scoped`), from the start when it names a source and once
    the pass's menu stands when it only excludes, each answer's identity binds the
    provenances that served it, and a no-data result carries the scope
    (`RegionResult.source_scope`). `source_code` is the StatVars the part's series code names
    (`search.engine.source_code`), which recall reads in place of the embedding resolve
    (`ResolveState.source_code`); such a part gets no UN pass (`_un_pass_skip`).
    `unit_text` carries a parsed-door request and the raw-text door's typed unit phrase. A bound, expressible unit narrows
    observation facets. With `unit_text_best_effort`, a unit the elected data cannot convert
    to is withdrawn and the leg answers unconverted (`phase_shared.withdraw_typed_unit`).
    `options` configures enrichment, streaming, clients, caches, and the deadline. `None` uses default `EngineOptions()` settings.
    `door` is the door the part was posted on, which decides who states each field of the
    part's frame (`ResolveState.frame`); a direct caller states them as the parsed door's does.

    `graph` and `sdmx` are shared, thread-safe clients. Dev-finance role-keyed observation and
    availability reads use `sdmx`. `llm` is stateless per call. `binder` reads the quantity
    and binds on `llm`. It is the request's own, which `core` routes once per request
    (`binders.configured_binder`) and every door passes down, or the Gemini binder for a
    direct caller that passes none. `date_request` is the shared window, while `base_timing`
    carries extraction timing.

    Recall runs once. Our pass elects over our candidates first, while the UN System Data
    Commons' resolve and shape derivation run beside it (`ResolveState.un_leg`). When our
    pass ends `no_data`, or answers on a pick its binder rated "none of the shapes" above,
    a UN pass elects over the UN pool's shapes (`_fill_gap`). Any other part answered on
    our pass never waits for the UN leg. A part whose UN leg failed or was late and that
    the UN pass could have changed carries `un_system_unavailable`, which keeps the
    response out of the response cache. With `SEARCH_UN_ON_FIRST_MENU`, a part holding no
    strong shape of its own waits for the UN leg and its menu also holds the UN pool's
    shapes (`phase_intake._merge_un_pool`); when the leg failed or was late, the part
    carries `un_system_unavailable` whatever its status. A part with a strong shape never takes the
    UN pool on its pass. A merged pass that ends `no_data` still runs the UN pass, and a
    doubted answer on a merged pass does not.

    Returns a `RegionResult` with status `definite`, `candidates`, `no_data`, or `unsupported`.
    When the pipeline's UN System Data Commons reads named places the UN front door refuses,
    the result carries `un_system_unavailable` naming them. The request's first subject-only
    no-data answer with an elected shape carries `withheld_by_registry` when the catalogue
    withholds the provenances holding its data (`region.withheld`). The result is `degraded`
    when the call swallowed a read (`degraded_reads`).
    """
    _admission = admission if admission is not None else Admission(graph=graph)
    _options = options if options is not None else EngineOptions()
    _binder = binder if binder is not None else GeminiBinder()

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
        pool: Pool,
        emit: EventSink | None,
        deadline: float | None,
        timing: dict[str, int],
        party_words_off: bool,
    ) -> ResolveState:
        return ResolveState(
            party_words_off=party_words_off,
            variable=variable,
            entities=entities,
            date_request=date_request,
            detect_query=detect_query,
            role_query=role_query,
            entity_roles=entity_roles or {},
            member_entities=member_entities,
            dropped_qualifier_phrases=dropped_qualifier_phrases or [],
            operations=tuple(operations),
            open_axes=open_axes,
            query_text=query_text,
            qualifiers=qualifiers,
            door=door,
            pool=pool,
            pac=pac,
            graph=graph,
            resolver=resolver,
            admission=_admission,
            llm=llm,
            binder=_binder,
            sdmx=sdmx,
            deadline=deadline,
            pre_resolved=pre_resolved,
            entity_surfaces=entity_surfaces or {},
            variable_index=variable_index,
            emit=emit,
            include_option_coverage=_options.include_option_coverage,
            stream_slot_options=_options.stream_slot_options,
            include_observations=_options.include_observations,
            stream_observations=_options.stream_observations,
            unit_text=unit_text,
            unit_text_best_effort=unit_text_best_effort,
            scope=source_scope,
            source_code=source_code,
            pin_served_provenance=source_scope is not None,
            materialise_graph=materialise_graph,
            materialise_sdmx=_options.materialise_sdmx,
            rate_cache=_options.rate_cache,
            population_memo=_options.population_memo,
            place_options_pool=_options.place_options_pool,
            role_class_walks=role_class_walks,
            coverage=coverage,
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

    async def attempt(
        *, party_words_off: bool, sink: EventSink | None, carried: Sequence[str] = ()
    ) -> tuple[RegionResult, tuple[str, ...]]:
        """One resolution of the part and the reads it swallowed, read with or without the
        words that name a party. `carried` are reads an abandoned resolution
        swallowed, noted on this one."""
        state = new_state(
            pool="ours",
            emit=sink,
            deadline=_options.deadline,
            timing=dict(base_timing),
            party_words_off=party_words_off,
        )
        # A read the pass swallows marks the part degraded, so no cache stores it. A part whose
        # scope names a source reads through the catalogue scoped to it throughout. One that only
        # excludes sources ranks on the evidence the unscoped question reaches
        # (`_phase_prune_and_rank`): its scope binds for recall, which reads it only to recall
        # in-scope families past the cut (`with_rival_families`), for each pass once its menu
        # stands (`_run_phases`), and for the withheld warning.
        allowed = None if source_scope is None else source_scope.allowed
        lost = DegradedReads()
        try:
            with (
                held_family_memo(),
                recording_degraded_reads() as lost,
                scoped_reads(allowed if source_scope is not None and source_scope.named else None),
            ):
                for read in carried:
                    note_degraded_read(read)
                with recording_un_blocked_places() as un_blocked:
                    enter_phase("recall")
                    with scoped_reads(allowed):
                        await _phase_recall(state)
                    try:
                        result = _within_source_code(
                            _excluded_from_the_place(await _run_phases(state), state), state
                        )
                        enter_phase("fill_gap")
                        result = await _fill_gap(
                            result,
                            first=state,
                            deadline=_options.deadline,
                            un_state=lambda bound: new_state(
                                pool="un_system",
                                emit=None,
                                deadline=bound,
                                timing={},
                                party_words_off=True,
                            ),
                        )
                    finally:
                        if state.un_leg is not None:
                            state.un_leg.task.cancel()
                result = with_un_unavailable_warning(result, un_blocked)
                # The withheld re-read runs once per request (`claim_withheld_read`) and serves
                # only a subject-only no-data answer (`withheld.elected_series`). A party
                # reading stands only when it answers with the party on its axis
                # (`binds_each_party`), so it never needs the re-read, and one it is withdrawn
                # from must leave the re-read to the reading that answers.
                if party_words_off:
                    with scoped_reads(allowed):
                        result = await with_withheld_warning(
                            result, state.graph, deadline=state.deadline
                        )
        except PartyReadingWithdrawn as exc:
            raise PartyReadingWithdrawn(lost=tuple(lost.reads)) from exc
        except Exception as exc:
            # A party reading that fails or refuses is withdrawn, whatever the cause: the
            # resolution without party words makes the reads and elections the base makes,
            # and fails or refuses as it does. Only a failed read marks the part degraded,
            # and any other exception than a failed read or a refusal is a defect.
            if party_words_off:
                raise
            failed_read = isinstance(exc, _UPSTREAM_FAILURES)
            if failed_read or isinstance(exc, _REQUEST_REFUSALS):
                logger.warning(
                    "the party reading of part %d (%r) withdrawn on %s: %s",
                    variable_index,
                    variable,
                    type(exc).__name__,
                    exc,
                )
            else:
                logger.exception(
                    "the party reading of part %d (%r) failed", variable_index, variable
                )
            raise PartyReadingWithdrawn(
                lost=(*lost.reads, "party reading") if failed_read else tuple(lost.reads),
            ) from exc
        if result.specs and state.scope is not None and state.scope.named:
            # An included source none of whose provenances the answer binds served nothing of
            # it, and only the read knows which provenances served.
            served = answer_provenances(result.specs)
            if served is not None:
                marked = with_idle_inclusions(state.scope, served)
                warnings = (*result.warnings, *idle_inclusion_warnings(marked))
                result = dataclasses.replace(result, warnings=warnings)
        if result.status == "no_data" and state.scope is not None:
            # Every no-data outcome of a scoped part, from candidacy, election or the read, names
            # the sources that limited it. A no-data result is our pass's, whose scope marks
            # the exclusions that removed nothing.
            result = dataclasses.replace(result, source_scope=state.scope)
        result = dataclasses.replace(result, degraded=True) if lost.degraded else result
        return result, tuple(lost.reads)

    # A part stating a word that names a party (`place_roles.stated_party_words`) is read
    # with it first. The answer stands only when every Spec binds the party on the word's
    # axis. Otherwise the part is resolved again without party words, which is the reading
    # of a request that states none, so nothing of the first reading reaches the reply: its
    # events are held back until it stands. The reply reports the model usage of both
    # readings, the first's read from every call it made wherever it ended
    # (`recording_llm_usage`), and notes the reads the first swallowed or failed.
    probe = new_state(
        pool="ours", emit=None, deadline=_options.deadline, timing={}, party_words_off=False
    )
    named = stated_party_words(probe)
    if not named:
        result, _lost = await attempt(party_words_off=True, sink=emit)
        return result
    held: list[Event] = []

    async def hold(event: Event) -> None:
        held.append(event)

    with recording_llm_usage() as spent:
        try:
            first, carried = await attempt(
                party_words_off=False, sink=None if emit is None else hold
            )
        except PartyReadingWithdrawn as withdrawn:
            carried = withdrawn.lost
        else:
            if binds_each_party(first, named):
                if emit is not None:
                    for event in held:
                        await emit(event)
                return first
    result, _lost = await attempt(party_words_off=True, sink=emit, carried=carried)
    usage = functools.reduce(_sum_usage, spent.calls, result.llm_usage)
    return dataclasses.replace(result, llm_usage=usage)


def binds_each_party(result: RegionResult, named: Sequence[PartyWord]) -> bool:
    """Whether `result` answers, and every Spec it holds binds the party of one of `named`
    alone on that word's axis."""
    return bool(result.specs) and all(
        any(spec.definition.identity.entities.get(word.axis) == [word.party] for word in named)
        for spec in result.specs
    )


def _within_source_code(result: RegionResult, state: ResolveState) -> RegionResult:
    """`result`, or the refusal of a part whose series code named its StatVars
    (`ResolveState.source_code`) and one of whose Specs mints an identity whose Datasets
    mint serves StatVars outside them, or ones the code index does not resolve in memory
    (`bound_values.elects_within`). The orchestrator fetches the Spec by its spec_id, so
    such an answer would serve another series than the code's."""
    code = state.source_code
    if code is None:
        return result
    for spec in result.specs:
        if not elects_within(spec.definition.identity, code.dcids):
            logger.info(
                "source code %s: spec %s serves StatVars outside the code's",
                code.code,
                spec.spec_id,
            )
            return state.no_data("variable_not_resolved", llm_usage=result.llm_usage)
    return result


def _excluded_from_the_place(result: RegionResult, state: ResolveState) -> RegionResult:
    """`result`, or the refusal of a part whose exclusions alone left its answer without
    data. A part that only excludes sources ranks on the evidence the unscoped question
    reaches, so derive's place probe read its elected shape's members unscoped
    (`ShapeDraft.member_coverage`). When that probe found an elected member with data at one
    of the places and the scoped read found no elected member with data at any of them
    (`withheld.elected_series`), the exclusions removed it. The part refuses
    ``qualifier_not_bound`` naming the excluded sources, which marks the scope as settling
    the answer (`phase_rank_bind.exclusion_records`), so the UN pass does not run. A shape
    the probe did not cover keeps `result`."""
    scope = state.scope
    series = elected_series(result)
    shape = state.shape_draft
    if scope is None or scope.named or series is None or shape is None:
        return result
    coverage = shape.member_coverage
    variables, places = series
    if coverage is None or not any(
        place in coverage.get(variable, {}) for variable in variables for place in places
    ):
        return result
    return dataclasses.replace(
        result,
        no_data_reason="qualifier_not_bound",
        gate=None,
        unbound_sources=exclusion_records(state, scope),
    )


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


async def _fill_gap(
    result: RegionResult,
    *,
    first: ResolveState,
    deadline: float | None,
    un_state: Callable[[float | None], ResolveState],
) -> RegionResult:
    """`result`, our pass's answer, or the UN pass's answer when the UN System Data Commons
    may fill a gap in it (`_un_pass_skip`): `result` is `no_data`, or it answers on a pick
    its binder rated "none of the shapes" above (`_doubted`).

    The UN pass runs the phases again over the shapes the UN leg derived beside our pass
    (`ResolveState.un_leg`), sharing recall, less the shape our pass elected when it found
    it without observations or answered on it doubted: a superseded variable stands in our
    pass as the UN variable the UN resolve may recall too, and the UN pass would elect it
    again. It waits for the leg no longer than `SEARCH_UN_FIRST_PASS_WAIT_S` from the start
    of our derivation, and ends `_UN_PASS_RESERVE_S` before `deadline`. Its answer stands
    when it is `definite` or `candidates`, unless both binders doubted their picks, or when
    both passes refuse and it carries an `alternate_match` and `result` carries none. Of
    two refusals, the one offering a reading with data at the place stands, as status rule
    1 has a nominee ride on a refusal, and so does a UN refusal on a variable it matched
    over ours `variable_not_resolved`: "no data for Japan" says more than "could not
    match". Otherwise `result` stands with its reason. A `result` holding an
    `alternate_match`, a reading the model nominated from our pool with data at the place,
    also stands when the UN pass's elected shape matches the phrase only weakly
    (`_representative_score_is_weak`): such an answer has no better claim to the question
    than the alternate. The answer carries both passes' step timings and model usage, and
    states `alternates_unevaluated` when either pass left an alternate unevaluated. The
    dropped answer's trailing tasks are cancelled (`deadline._cancel_trailing_tasks`).

    A UN leg that failed, or was not in by its wait, and a UN pass that fails upstream
    (`_UPSTREAM_FAILURES`) or has no time to start, leave `result` standing with
    `un_system_unavailable`, only ever on an answer the UN pool could have changed: a
    `no_data` one or a doubted one. A leg our pass waited for and missed
    (`ResolveState.un_pool`) is the exception: the menu lacked the pool, so every answer
    carries the warning and no UN pass runs. A pool merged onto our pass's menu is the UN
    pass's pool when ours ends `no_data`. What the leg's reads could not serve
    (`UnPool.unserved`, `UnPool.lost`) joins the part's records only when the pool reaches
    a menu, so an answer our pass serves without it carries none of it. A declared-default refusal
    (`DeclaredDefaultUnsupportedError`), an index-sourced shape meeting a default scope
    (`DefaultScopeIndexSourcedError`) or a conversion-table refusal (`RATE_TABLE_REFUSALS`)
    in the UN pass propagates, since each refuses the request on every door; any other
    failure is a defect, logged, and leaves `result` standing. Cancellation propagates. The UN pass sends no events: the frame, shape and
    step events on the stream describe our pass.
    """
    bound = None if deadline is None else deadline - _UN_PASS_RESERVE_S
    merged = first.un_pool
    if merged is not None and merged.missed is not None:
        _log_gap_fill(first, result, un_pass="skipped:un_unavailable")
        return _with_un_search_warning(result, merged.missed)
    skip = _un_pass_skip(result, first)
    leg = first.un_leg
    if skip is not None or leg is None:
        # A part that ended before derivation started no UN leg.
        _log_gap_fill(first, result, un_pass=f"skipped:{skip or 'no_un_leg'}")
        return result
    pool = merged if merged is not None else await un_pool_shapes(leg, bound=bound)
    if pool.missed is not None:
        _log_gap_fill(first, result, un_pass="skipped:un_unavailable")
        return _with_un_search_warning(result, pool.missed)
    shapes = pool.shapes
    elected = first.shape_draft
    doubted = _doubted(result, first)
    if (result.no_data_reason == "no_observations" or doubted) and elected is not None:
        read = set(elected.sv_arc_facts or {})
        shapes = [
            shape
            for shape in shapes
            if shape.shape_id != elected.shape_id or not set(shape.sv_arc_facts or {}) <= read
        ]
    if not shapes:
        _log_gap_fill(first, result, un_pass="skipped:no_un_shapes")
        return result
    if bound is not None and time.monotonic() >= bound:
        _log_gap_fill(first, result, un_pass="skipped:deadline")
        return _with_un_search_warning(result, _UN_PASS_INCOMPLETE)

    # The leg's reads bear on the answer only now the UN pass runs over its shapes, unless
    # our pass's menu already held them.
    if merged is None:
        note_un_unserved(pool.unserved)
        for read in pool.lost:
            note_degraded_read(read)
    state = un_state(bound)
    state.rcl = first.rcl
    state.shapes_found = shapes
    t0 = now_ms()
    try:
        async with asyncio.timeout(None if bound is None else bound - time.monotonic()):
            second = await _run_phases(state)
    except Exception as exc:
        _log_gap_fill(
            first,
            result,
            un_pass=f"error:{type(exc).__name__}",
            un_ms=now_ms() - t0,
        )
        if isinstance(exc, _REQUEST_REFUSALS):
            raise
        if isinstance(exc, _UPSTREAM_FAILURES):
            return _with_un_search_warning(result, _UN_PASS_INCOMPLETE)
        logger.exception("the UN pass of part %d failed", first.variable_index)
        return result
    un_elected = state.shape_draft
    _log_gap_fill(
        first,
        result,
        un_pass=f"ran:{second.status}/{second.no_data_reason}",
        un_ms=now_ms() - t0,
    )
    displaces_alternate = result.alternate_match is None or not _representative_score_is_weak(
        None if un_elected is None else un_elected.measured_score
    )
    un_doubted = state.bind_decision is not None and state.bind_decision.none_first
    answers = (
        second.status in ("definite", "candidates")
        and displaces_alternate
        and not (doubted and un_doubted)
    )
    refused = result.status == "no_data" and second.status == "no_data"
    # Of two refusals, the one carrying a nominee with data stands (status rule 1).
    offers_alternate = (
        refused and second.alternate_match is not None and result.alternate_match is None
    )
    # A UN refusal that matched a variable says more than ours that matched none. Every UN
    # reason but `variable_not_resolved` is reached after election: the pass shares our
    # pass's sources and entities, which got ours as far as election, and mints
    # `entity_set_too_large` and `entity_role_not_admitted` only on its elected shape's
    # places, though `NO_DATA_RESOLUTION_ORDER` ranks both before the variable.
    matched_further = (
        refused
        and result.no_data_reason == "variable_not_resolved"
        and second.no_data_reason != "variable_not_resolved"
    )
    chosen = second if answers or offers_alternate or matched_further else result
    timing = dict(result.timing_by_step)
    for step, ms in second.timing_by_step.items():
        timing[step] = timing.get(step, 0) + ms
    other = result if chosen is second else second
    # The dropped answer's trailing reads serve a Spec the response never names.
    await _cancel_trailing_tasks([other])
    carried = (
        ()
        if chosen.alternates_left_out
        else tuple(w for w in other.warnings if w.code == ALTERNATES_UNEVALUATED)
    )
    return dataclasses.replace(
        chosen,
        warnings=(*chosen.warnings, *carried),
        timing_by_step=timing,
        llm_usage=_sum_usage(result.llm_usage, second.llm_usage),
        scope_unsettled=result.scope_unsettled or second.scope_unsettled,
    )


def _doubted(result: RegionResult, first: ResolveState) -> bool:
    """Whether our pass's `result` answers on a pick its binder rated "none of the shapes"
    above (`BindResult.none_first`), on the resolution without party words
    (`ResolveState.party_words_off`). A UN answer binds no party, so on the resolution with
    them it could only fail `binds_each_party` and send the part to the other one."""
    decision = first.bind_decision
    return (
        result.status in ("definite", "candidates")
        and first.party_words_off
        and decision is not None
        and decision.none_first
    )


def _un_pass_skip(result: RegionResult, first: ResolveState) -> str | None:
    """Why the UN pass does not run after our pass's `result`, or `None` when it runs.

    Only a `no_data` answer has a gap to fill, and an answer its binder doubted
    (`_doubted`) one the UN pool may fill better, unless our pass's menu already held the
    UN pool (`ResolveState.un_pool`): the binder rated "none" above those shapes too, and a
    UN pass would only elect again among them. A refusal about the request's entities or a
    `source_ambiguous` one stands, since the UN pool resolves neither. So does the answer of
    a part whose series code named its StatVars (`ResolveState.source_code`), which answers
    from those alone, and that of a part whose
    scope allows no UN System Data Commons provenance (`source_choice.allows_un_system`),
    since a UN answer would serve a source the part left out, or a part that only excludes
    sources and refused ``qualifier_not_bound`` on its exclusion
    (`ResolveState.scope_settled`): our catalogue holds the measure and the part left its
    publishers out. A part that names a source and refuses on its scope is not settled by
    the refusal; whether its UN pass runs is `allows_un_system`'s to say, so a part naming a
    UN System Data Commons source still reaches it."""
    if result.status != "no_data":
        if not _doubted(result, first):
            return "answered"
        if first.un_pool is not None:
            return "merged"
    if result.no_data_reason in _ENTITY_TERMINALS:
        return "entity"
    if result.no_data_reason == "source_ambiguous":
        return "source_ambiguous"
    if first.source_code is not None:
        return "source_code"
    if first.scope_settled or not allows_un_system(first.scope):
        return "source_scope"
    return None


def _with_un_search_warning(result: RegionResult, message: str) -> RegionResult:
    warning = Warning(code=UN_SYSTEM_UNAVAILABLE, severity="warn", message=message)
    return dataclasses.replace(result, warnings=(*result.warnings, warning))


def _log_gap_fill(
    first: ResolveState, result: RegionResult, *, un_pass: str, un_ms: int = 0
) -> None:
    """One line per part: our pass's status, the pool of the shape it elected
    (`ShapeDraft.pool`, `-` when it elected none), what the UN pass did (why it was
    skipped, or the status it reached and its time), whether our pass's bind asked
    (`_UniversalBindOutput.ask`), which its status alone does not show once a declined
    reading or a source-named term refuses in its place, whether our pass's menu held
    the UN pool (`ResolveState.un_pool`), and whether its binder doubted its answer
    (`_doubted`), which runs the UN pass on an answered part."""
    elected = first.shape_draft
    bind = first.bind_result
    merged = first.un_pool
    logger.info(
        "gap_fill part=%d status=%s/%s pool=%s un_pass=%s un_ms=%d bind_asked=%s un_merged=%s "
        "doubted=%s",
        first.variable_index,
        result.status,
        result.no_data_reason,
        "-" if elected is None else elected.pool,
        un_pass,
        un_ms,
        "true" if bind is not None and bind.ask is not None else "false",
        "true" if merged is not None and merged.missed is None else "false",
        "true" if _doubted(result, first) else "false",
    )


async def _run_phases(state: ResolveState) -> RegionResult:
    """Run `_PHASES` in order until one returns the terminal result. Once the pass's menu
    stands (`ResolveState.capped_shapes`), every later read goes through the catalogue scoped
    to the part's sources (`ResolveState.scope`), so a part that only excludes sources ranks
    unscoped and reads scoped from election on. However the pass ends, it then cancels what
    it queued (`_cancel_queued_reads`) and ends its related-measures reads
    (`_end_related_reads`). Each phase's time that no step booked lands under
    `phase_<name>` (`_book_unbooked_time`)."""
    scoped = False
    settled = False
    unbooked: dict[str, int] = {}
    try:
        with ExitStack() as reads:
            for phase in _PHASES:
                # Every `_PHASES` entry is a plain function, but `Callable` hides its `__name__`.
                name = phase.__name__.removeprefix("_phase_")  # ty: ignore[unresolved-attribute]
                enter_phase(f"{state.pool}:{name}")
                t0, booked_before = now_ms(), _booked_ms(state)
                result = await phase(state)
                # A refusal once the bindings are final that judged no answer is judged
                # on the elected answer, so each requirement is named or departs
                # (`settle_unjudged_refusal`).
                if (
                    result is not None
                    and settled
                    and state.judgement is None
                    and result.status in ("no_data", "unsupported")
                ):
                    result = await settle_unjudged_refusal(state, result)
                _book_unbooked_time(
                    state,
                    unbooked,
                    f"phase_{name}",
                    elapsed=now_ms() - t0,
                    booked=_booked_ms(state) - booked_before,
                )
                if result is None:
                    settled = settled or phase is _phase_settle
                else:
                    # Only a vetted election can report qualifier matches: earlier terminals
                    # may hold pre-bind candidates without a forced binding or elected shape.
                    admitted = (
                        _admitted_qualifier_phrases(state.admitted_qualifiers)
                        | state.satisfied_measure_phrases
                        | state.elected_tie_phrases
                        | state.applied_dropped_qualifier_phrases
                        | unbinding_phrases(state.requirements)
                        | state.slot_default_phrases
                        if state.shape_draft is not None
                        else _refused_pinned_phrases(state, result)
                    )
                    # Every requirement departs here, once, as the judgement of the part's
                    # final answer leaves it (`with_requirement_departures`).
                    return _stating_unevaluated(
                        dataclasses.replace(
                            with_requirement_departures(
                                result, state.requirements, state.judgement
                            ),
                            admitted_qualifier_phrases=admitted,
                            timing_by_step={**result.timing_by_step, **unbooked},
                            scope_unsettled=_scope_unsettled(state),
                            entity_words=group_request_names(state),
                            inferred_facts=(
                                tuple(state.inferred_facts) if result.status == "definite" else ()
                            ),
                        ),
                        state,
                    )
                if state.capped_shapes and state.scope is not None and not scoped:
                    reads.enter_context(scoped_reads(state.scope.allowed))
                    scoped = True
        raise AssertionError("_phase_outcome is terminal")
    finally:
        _cancel_queued_reads(state)
        enter_phase(f"{state.pool}:end_related_reads")
        await _end_related_reads(state)


def _booked_ms(state: ResolveState) -> int:
    """The ms `state.timing` books, leaving out the stages inside `shape`
    (`conjoin.NESTED_SHAPE_STEPS`), since `shape` already spans them."""
    return sum(ms for step, ms in state.timing.items() if step not in NESTED_SHAPE_STEPS)


def _book_unbooked_time(
    state: ResolveState,
    unbooked: dict[str, int],
    step: str,
    *,
    elapsed: int,
    booked: int,
) -> None:
    """Book under `step` the part of a phase's `elapsed` ms that its own steps (`booked`)
    left out, so no millisecond is counted twice. A step that books from before its phase
    began (`shape` starts in `_phase_derive_shapes` and ends in `_phase_prune_and_rank`)
    books more than the phase ran; the excess comes off the latest earlier phases
    already booked in `unbooked`, whose entries `state.timing` mirrors."""
    excess = booked - elapsed
    for earlier in reversed(unbooked):
        if excess <= 0:
            break
        taken = min(unbooked[earlier], excess)
        unbooked[earlier] -= taken
        state.timing[earlier] -= taken
        excess -= taken
    unbooked[step] = state.timing[step] = max(0, elapsed - booked)


def _refused_pinned_phrases(state: ResolveState, result: RegionResult) -> frozenset[str]:
    """The phrases of the codes a refusal before bind names as pinned codes no reading of
    the part can bind (`admissibility.judge_pinned_parts`). The refusal states each one, so
    no dropped-qualifier warning repeats it."""
    if result.no_data_reason != "qualifier_not_bound":
        return frozenset()
    return frozenset(
        match.phrase
        for interpretation in state.interpretations.interpretations
        for match in interpretation.pinned_codes
    )


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
# available earlier, so the first terminal is the earliest applicable reason, with one
# exception, `entity_set_too_large` for an elected shape, below.
# `source_not_resolved` and `source_ambiguous` come first, then the part's series code refusals.
# A part's source mentions and series code compile at its door, and a refusal answers the leg
# before any phase runs (`fanout._stream_one_variable`). A held code's part mints
# `source_code_not_held` at every phase that would mint `variable_not_resolved`
# (`ResolveState.no_data`), so it ranks at the variable position there.
# Entity errors come from `_phase_check_entities`, which runs before shape derivation and
# the bind call. Each name with no place set mints `entity_not_resolved`, `entity_ambiguous`
# (an Ambiguous resolution) or `entity_set_too_large` (a Group with more members than a
# Spec binds), and `elect_no_data_reason` picks the part's reason from them.
# `_phase_place_bindings` mints `entity_set_too_large` too, for an
# elected shape whose places exceed that bound. The count depends on the elected shape's
# roles, so it is known only once a shape is elected. A request whose variable does not
# resolve answers `variable_not_resolved` even when its places exceed the bound.
# `_phase_derive_shapes` mints `entity_role_not_admitted` when each shape fails to admit at
# least one resolved request entity. It mints `role_signature_incompatible` for
# mismatched signatures across candidate groups.
# `variable_not_resolved` comes from derive_shapes (no confirmed shape), prune/rank (no eligible
# shape), or bind (an ask, an out-of-range election, or an answer the schema refuses at parse,
# a binding that names no value).
# `_phase_place_bindings` mints `role_not_resolved` for unresolved multi-place axes, a solitary
# place without a supplied axis, or contradicted wording. It mints `role_incompatible` for a
# directional query against a non-role shape or an unplaceable role entity.
# Materialise can also mint `role_not_resolved` when declared roles miss destination vocabulary.
# `_phase_facet_requests` handles `unit_not_resolved` when a bound unit fails parsing. It also
# rejects inexpressible units as unsupported.
# `qualifier_not_bound` comes from prune/rank for a source scope that leaves no recalled
# shape; from `_phase_bind` for pinned codes no reading can bind (`_refusal_before_bind`),
# code terms no offered shape holds (`_unelected_qualifiers`) and sources whose readings
# none inside the scope accepts (`_declined_terminal`); from `_phase_settle` for codes or
# a row the part's tables do not hold, which judges no answer and names the ledger's
# requirements only an answer can meet (`requirements.settle_unanswered`); from
# `_phase_materialise` for a declined reading's pinned codes (`_declined_refusal`); and
# from definite assembly after identity confirmation or candidates' single-survivor
# collapse. The last two call `requirements.settle_answer`, so each value bind drafted on
# a slot the elected shape lacks is among the terms they name. A later refusal that judged
# no answer and whose reason ranks after `qualifier_not_bound` gives way to it when the
# elected answer leaves terms unmet (`phase_shared.settle_unjudged_refusal`).
# Materialise can mint `no_matching_slice`, deferred to the place-less mint, or `binding_mismatch`.
# `_phase_outcome` forwards materialised `binding_mismatch` reasons.
# `_phase_materialise` reports `denominator_not_available` after `no_matching_slice` is resolved.
# `no_observations` and `facet_not_available` come from `_phase_outcome`: the elected
# shape's no-data read, candidates, grounding's window answer, or definite assembly for the
# first; candidates or assembly for an unmatched unit facet, or grounding before fetch when
# no convertible facet can be selected, for the second.
# `resolve_variable` runs `_phase_recall` once, before the phases.
_PHASES: tuple[Callable[[ResolveState], Awaitable[RegionResult | None]], ...] = (
    _phase_resolve_places,
    _phase_check_entities,
    _phase_derive_shapes,
    _phase_prune_and_rank,
    _phase_bind,
    _phase_settle,
    _phase_place_bindings,
    _phase_facet_requests,
    _phase_materialise,
    _phase_outcome,
)
