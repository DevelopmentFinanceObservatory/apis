"""Compute raw-text signals and coordinate the two-layer response cache and its part memo.

`CacheScope` sequences lookups, writes, and release for eligible raw-text and parsed
requests; callers do not re-check its `active` conditions. It looks up layer 1 before
layer 2, preventing a lease-order cycle. `PartMemoScope` serves a `dataset_resubmit`
request's identity entries from the part memo.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import asdict, dataclass

import search.engine.config as config  # module-attribute access for runtime-resolved values
from code_index.shared import get_index, get_measure_index
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from origin_catalogue import current_catalogue
from search.engine.bind_repair import (
    declared_default_slots_named,
    declared_default_words_named,
    stated_words_negated,
)
from search.engine.binders import Binder, Fallbacks
from search.engine.deadline import _await_bounded
from search.engine.enrichment import stream_place_options
from search.engine.envelope import _terminal_events, now_ms
from search.engine.events import DoneEvent, Event, PlaceOptionsEvent, ResponseEvent
from search.engine.extract import ExtractedDate, ExtractedEntity
from search.engine.qualifier_match import (
    CodedQualifier,
    match_qualifier_readings,
    match_qualifier_ties,
    match_qualifiers,
    match_unsupported_qualifier_phrases,
    qualifier_occurrences,
)
from search.engine.region.promote_relaunch import without_relaunches
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.response_cache import Lease, ResponseCache
from search.engine.response_cache_keys import (
    StoredBundle,
    from_bundle,
    from_entry,
    is_bundle_storable,
    is_region_storable,
    is_storable,
    layer1_key,
    layer2_key,
    parsed_key,
    part_memo_key,
    refresh_response,
    scope_document,
    to_bundle,
    to_entry,
)
from search.engine.response_cache_memo import PartMemo
from search.engine.role_text import DoorRole, contradicts, every_member_axis
from search.engine.stated_scope import intake_key_document
from search.engine.unit.conversion_catalogue import current_conversion_catalogue
from search.engine.written_phrase import written_phrase
from search.models import (
    Operation,
    ParsedInput,
    ResolveRequest,
    ResolveResponse,
    ResubmitIdentityPayload,
    RoleAxis,
    SourceMention,
    Warning,
)
from search.scope.state import ScopeIntake
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

type ReplayedPlaceOptions = Callable[
    [ResolveResponse], Sequence[asyncio.Task[list[PlaceOptionsEvent]]]
]
"""Launches the place-options tasks a fresh run of a response launches
(`place_options.replayed_place_options`, with the request's clients bound)."""


# ---------------------------------------------------------------------------
# Raw-text inputs absent from Extraction: two `role_query` readers run on every leg. One
# checks each DoorRole occurrence, the other reads the flow axis each dropped phrase asks
# every member of (`role_text.every_member_axis`). Single-variable requests also read
# qualifier vocabulary, defaults, written phrases, negated dataset words and graph value
# names.
# `test_query_text_and_role_query_reader_counts_are_covered` guards this set.
# Graph value names are known only after retrieval, so their signal keeps the whole
# query. The other signals reuse the pipeline helpers and index.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layer2Signals:
    """JSON-safe raw-text signals combined with `Extraction` in the layer-2 key.

    Different queries share a key when their effective extractions and signals match.
    """

    role_contradictions: tuple[dict[str, object], ...]
    query_text_qualifier_matches: tuple[dict[str, object], ...]
    query_text_qualifier_ties: tuple[dict[str, object], ...]
    query_text_qualifier_readings: tuple[dict[str, object], ...]
    query_text_qualifier_occurrences: tuple[dict[str, object], ...]
    unsupported_qualifier_phrases: tuple[str, ...]
    declared_default_trigger_words: tuple[dict[str, str], ...]
    declared_default_slots_named: tuple[str, ...]
    written_variable_phrases: tuple[str | None, ...]
    query_text_negated_stated_words: tuple[str, ...]
    every_member_axes: tuple[str | None, ...]
    query_text_value_source: str | None


def _role_contradiction(name: str, role: DoorRole, *, query: str) -> dict[str, object]:
    # Include the entity name: distinct names can share a span and axis, but
    # the pipeline checks each name's role occurrences separately.
    return {
        "name": name,
        "span": role.span,
        "axis": role.axis,
        "contradicts": contradicts(query, span=role.span, axis=role.axis),
    }


def compute_layer2_signals(
    *,
    query: str,
    entity_roles: Mapping[str, Sequence[DoorRole]],
    query_text: str | None,
    variables: Sequence[str] = (),
    dropped_qualifiers: Sequence[str] = (),
) -> Layer2Signals:
    """`query` supplies every leg's raw role signals; `entity_roles` lists its
    door-provided occurrences in mention order, and each of `dropped_qualifiers` gives the
    flow axis it asks every member of, read with the preposition `query` puts before it.
    `query_text` is set only for a single-variable request, matching the pipeline's own
    raw-text readers. Each of
    `variables` gives the stretch of `query_text` that writes it (`written_phrase`), which
    the named-member binding reads.
    """
    role_contradictions = tuple(
        _role_contradiction(name, role, query=query)
        for name, roles in entity_roles.items()
        for role in roles
    )
    every_member_axes = tuple(
        every_member_axis(phrase, query=query) for phrase in dropped_qualifiers
    )
    if query_text is None:
        return Layer2Signals(
            role_contradictions=role_contradictions,
            query_text_qualifier_matches=(),
            query_text_qualifier_ties=(),
            query_text_qualifier_readings=(),
            query_text_qualifier_occurrences=(),
            unsupported_qualifier_phrases=(),
            declared_default_trigger_words=(),
            declared_default_slots_named=(),
            written_variable_phrases=(),
            query_text_negated_stated_words=(),
            every_member_axes=every_member_axes,
            query_text_value_source=None,
        )
    index = get_index()
    measure_index = get_measure_index()
    matches = match_qualifiers(query_text, index, measure_index)
    ties = match_qualifier_ties(query_text, index, measure_index)
    readings = match_qualifier_readings(query_text, index, measure_index)
    return Layer2Signals(
        role_contradictions=role_contradictions,
        query_text_qualifier_matches=tuple(asdict(match) for match in matches),
        query_text_qualifier_ties=tuple(asdict(tie) for tie in ties),
        query_text_qualifier_readings=tuple(asdict(reading) for reading in readings),
        query_text_qualifier_occurrences=qualifier_occurrences(query_text, index, measure_index),
        unsupported_qualifier_phrases=match_unsupported_qualifier_phrases(
            query_text,
            terms=frozenset(
                term
                for declaration in index.declarations.values()
                for term in declaration.no_vocabulary_terms
            ),
        ),
        declared_default_trigger_words=tuple(
            {"code_system": code_system, "word": word}
            for code_system, word in declared_default_words_named(query_text, index=index)
        ),
        declared_default_slots_named=declared_default_slots_named(query_text, index=index),
        written_variable_phrases=tuple(
            written_phrase(variable, query_text) for variable in variables
        ),
        query_text_negated_stated_words=stated_words_negated(query_text, index=index),
        every_member_axes=every_member_axes,
        query_text_value_source=query_text,
    )


# What a hit spends: no model call runs.
_NO_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


class CacheScope:
    """One request's access to the two-layer response cache.

    `active` is fixed at construction; inactive methods return `None`. A raw-text or
    parsed request is active when the cache is enabled. A request that streams trailing
    events (`trailing`) stores its event bundle: the `response` event, the slot-options and
    observations events and the `done` event, published when `done` arrives (`observe`).
    Place options stay out of it: they run on a best-effort budget and the options cache
    keeps them, so a hit runs them again (`hit_events`). Any other request stores its
    response at the `response` event. A hit sends either entry through `hit_events`. A
    follower of an in-flight streamed leader therefore waits
    for the leader's `done` before it sends anything. The generation stamp is read once
    and sets both the key and entry lifetime. New stamps produce new keys, leaving old
    entries unreachable. `binder` is the binder the request was routed to. Its name and
    model join the key, so answers two binders give never share one. Layer 2 serves one query's answer to any query with the same
    extraction, which holds only while the bind reads the extraction alone: a binder that
    reads the query's own text (`Binder.reads_query_text`) skips the extraction-keyed
    lookup (`lookup_parsed`), and layer 1, which keys on the query, serves its repeats.
    `fallbacks` is the request's record of calls its binder served from a fallback
    (`binders.recording_fallbacks`). A request that holds any stores nothing, since its
    answer is not the one its key names. The caller's
    coverage snapshot contributes its stamp to the same scope; this class does
    not read the coverage service again. The revision of the conversion catalogue and the
    build and UN System setting of the origin catalogue the request bound join the scope
    too, so an answer computed under one catalogue is never served once another is live.
    Cache clearing belongs to app wiring.
    """

    def __init__(
        self,
        cache: ResponseCache | None,
        *,
        request: ResolveRequest,
        start_ms: int,
        deadline: float,
        build_commit: str,
        binder: Binder,
        fallbacks: Fallbacks,
        seed: int | None,
        today: str,
        max_candidates: int,
        seam_default: bool,
        coverage: CoverageSnapshot,
        trailing: bool,
        generation: GenerationProvider = UNKNOWN_GENERATION,
    ) -> None:
        self._cache = cache
        self._fallbacks = fallbacks
        self._reads_query_text = binder.reads_query_text
        self._start_ms = start_ms
        self._deadline = deadline
        # Captured once, here, never re-read: a computation that started
        # under this stamp publishes under this stamp's key even if the
        # generation moves on mid-request (see graph_generation.stamp's own
        # docstring). `_stamp.key` folds into `_scope` below, and
        # `_stamp.max_age_s` becomes every entry this request's `store()`
        # writes (see that method).
        self._stamp = generation.stamp()
        # The catalogues this request's scope bound, which every answer it computes reads.
        conversion = current_conversion_catalogue()
        origin = current_catalogue()
        self.active = (
            cache is not None
            and config.settings().response_cache_enabled
            and request.input.kind in ("raw_text", "parsed")
        )
        self._bundle = trailing
        # The bound build's refused places, which the key's `origin_catalogue` names.
        self._refused_at_build = origin.un_blocked_places
        self._scope = (
            scope_document(
                schema_version=request.schema_version,
                options=request.response_options,
                max_candidates=max_candidates,
                seam_default=seam_default,
                qualifier_coverage=config.settings().qualifier_coverage,
                stated_word_bar=config.settings().stated_word_bar,
                build_commit=build_commit,
                binder=binder.name,
                model=binder.model,
                engine_model=config.settings().engine_model,
                seed=seed,
                today=today,
                graph_generation=self._stamp.key,
                coverage_stamp=coverage.stamp,
                conversion_catalogue=conversion.revision if conversion is not None else None,
                origin_catalogue=origin.build_id,
                un_system_enabled=origin.un_system_enabled,
                bundle=trailing,
            )
            if self.active
            else None
        )
        self._layer1_lease: Lease | None = None
        self._layer2_lease: Lease | None = None
        self._layer2_key: str | None = None
        self._timed_out = False
        self._degraded_leg = False
        self._trailing_incomplete = False
        self._served_timed_out = False
        self._served_storable = True
        # Events from `response` on, while this request records its bundle.
        self._recorded: list[Event] = []
        # The stored bundle a hit replays.
        self._replay: StoredBundle | None = None
        self._outcome: str = "inactive" if not self.active else "miss"

    @property
    def outcome(self) -> str:
        """Return this request's `inactive`, `miss`, `hit layer1`, or `hit layer2` result.

        Per-request state avoids inferring outcomes from concurrent global counters.
        """
        return self._outcome

    @property
    def served_timed_out(self) -> bool:
        """Whether a cache hit came from a leader whose run timed out.

        The caller reports this in `done.timed_out`; restored diagnostics omit
        it. Stored hits return `False`. `store()` also uses the leader's
        `_served_storable` verdict when republishing a hit.
        """
        return self._served_timed_out

    async def _acquire(self, key: str) -> tuple[bytes | None, Lease | None]:
        """Return `key`'s value, or `None` with the leader lease this request now holds.

        A follower waits for its leader within this request's deadline and takes the
        leader's verdict with its value. If the deadline expires, the caller proceeds
        unleased (`(None, None)`) and cannot publish this key. If the leader abandons,
        the caller re-bids and waits again under the same deadline, so one concurrent
        caller still leads rather than every one of them redoing the pipeline.
        """
        assert self._cache is not None
        value, lease = await self._cache.acquire(key)
        if value is not None:
            return value, None
        assert lease is not None  # acquire() always returns a lease alongside a miss
        while not lease.is_leader:
            try:
                awaited = await _await_bounded(
                    self._cache.await_leader(lease), deadline=self._deadline
                )
            except TimeoutError:
                return None, None  # this request's own deadline is spent; proceed unleased
            if awaited is not None:
                value, self._served_timed_out, self._served_storable = awaited
                return value, None
            value, lease = await self._cache.acquire(key)
            if value is not None:
                return value, None
            assert lease is not None
        return None, lease

    async def lookup_raw(self, query: str) -> ResolveResponse | None:
        """Look up layer 1 by normalised query.

        A hit restores the response; the first miss gets the leader lease
        (`_acquire`).
        """
        if not self.active:
            return None
        assert self._scope is not None
        value, self._layer1_lease = await self._acquire(layer1_key(query=query, scope=self._scope))
        return self._hit(value, query=query) if value is not None else None

    def _hit(self, value: bytes, *, query: str) -> ResolveResponse:
        self._outcome = "hit layer1"
        return self._restore(
            value,
            query=query,
            extract_skipped=True,
            # A hit spends zero tokens; `None` means the model reported no usage metadata.
            llm_usage=dict(_NO_USAGE),
            step="response_cache_layer1",
        )

    async def lookup_parsed(
        self,
        *,
        variables: Sequence[str],
        entities: Sequence[ExtractedEntity],
        dates: Sequence[ExtractedDate],
        dropped_qualifiers: Sequence[str],
        variable_qualifiers: Sequence[tuple[CodedQualifier, ...]],
        operations: Sequence[tuple[Operation, ...]],
        query: str,
        entity_roles: Mapping[str, Sequence[DoorRole]],
        sources: Sequence[tuple[SourceMention, ...]],
        query_text: str | None,
        source_codes: Sequence[str | None] = (),
        unit_texts: Sequence[str | None] = (),
        separate: Sequence[bool] = (),
        extraction_warnings: Sequence[Warning] = (),
        measure_qualifiers: Sequence[tuple[str, ...]] = (),
        open_axes: Sequence[frozenset[RoleAxis]] = (),
        stated_scopes: Sequence[ScopeIntake] = (),
        llm_usage: dict[str, int] | None = None,
    ) -> ResolveResponse | None:
        """Look up layer 2 by effective extraction and raw-text signals.

        `entity_roles`, `query_text`, and the parallel `variable_qualifiers`, `operations`
        and `sources` are required to build a complete key. `source_codes` runs parallel to
        `variables` and holds the series code the query writes for each. `unit_texts` runs
        parallel to `variables` and holds the unit text extraction peeled from each
        (`ExtractionResult.unit_texts`). `separate` runs parallel to
        `variables` and marks each one the raw query compares with another.
        `measure_qualifiers` runs parallel to `variables` and holds the words narrowing each
        one's measure. `open_axes` runs parallel to `variables` and holds the flow roles each
        one asks one value per member of. `stated_scopes` runs parallel to `variables`
        and holds the scope entries and total intent each one's words state.
        `extraction_warnings` holds the warnings extraction adds to the answer, which a hit
        serves as stored. This lookup
        never waits on a layer-1 lease. The first miss owns the layer-2 lease (`_acquire`).
        A binder that reads the query's own text looks up nothing here (`CacheScope`).

        Hits record the leader's health verdict. `store()` can then cache the
        request's literal query at layer 1 using that verdict; layer-2 hits
        never republish the layer-2 key.
        """
        if not self.active or self._reads_query_text:
            return None
        assert self._scope is not None
        signals = compute_layer2_signals(
            query=query,
            entity_roles=entity_roles,
            query_text=query_text,
            variables=variables,
            dropped_qualifiers=dropped_qualifiers,
        )
        key = layer2_key(
            variables=variables,
            entities=entities,
            dates=dates,
            dropped_qualifiers=dropped_qualifiers,
            variable_qualifiers=variable_qualifiers,
            operations=operations,
            sources=sources,
            source_codes=source_codes,
            unit_texts=unit_texts,
            role_contradictions=signals.role_contradictions,
            query_text_qualifier_matches=signals.query_text_qualifier_matches,
            query_text_qualifier_ties=signals.query_text_qualifier_ties,
            query_text_qualifier_readings=signals.query_text_qualifier_readings,
            query_text_qualifier_occurrences=signals.query_text_qualifier_occurrences,
            unsupported_qualifier_phrases=signals.unsupported_qualifier_phrases,
            declared_default_trigger_words=signals.declared_default_trigger_words,
            declared_default_slots_named=signals.declared_default_slots_named,
            written_variable_phrases=signals.written_variable_phrases,
            query_text_negated_stated_words=signals.query_text_negated_stated_words,
            every_member_axes=signals.every_member_axes,
            query_text_value_source=signals.query_text_value_source,
            scope=self._scope,
            separate=separate,
            extraction_warnings=extraction_warnings,
            measure_qualifiers=measure_qualifiers,
            open_axes=open_axes,
            stated_scopes=[intake_key_document(intake) for intake in stated_scopes],
        )
        self._layer2_key = key
        value, self._layer2_lease = await self._acquire(key)
        if value is None:
            return None
        return self._hit_layer2(value, query=query, llm_usage=llm_usage)

    async def lookup_parsed_input(
        self, parsed: ParsedInput, *, stated_scopes: Sequence[ScopeIntake] = ()
    ) -> ResolveResponse | None:
        """Look up layer 2 by a parsed request's input (`parsed_key`) and the scope each
        part is posted with, `stated_scopes`, in part order (`stated_scope.part_scope_intakes`).

        A hit skips binding, keeps the stored echo and spends no
        tokens. The first miss owns the layer-2 lease (`_acquire`).
        """
        if not self.active:
            return None
        assert self._scope is not None
        value, self._layer2_lease = await self._acquire(
            parsed_key(
                parsed=parsed,
                scope=self._scope,
                stated_scopes=[intake_key_document(intake) for intake in stated_scopes],
            )
        )
        if value is None:
            return None
        return self._hit_layer2(value, query=None, llm_usage=dict(_NO_USAGE))

    def _hit_layer2(
        self, value: bytes, *, query: str | None, llm_usage: dict[str, int] | None
    ) -> ResolveResponse:
        self._outcome = "hit layer2"
        return self._restore(
            value,
            query=query,
            extract_skipped=False,
            llm_usage=llm_usage,
            step="response_cache_layer2",
        )

    def note_health(
        self,
        *,
        timed_out: bool = False,
        degraded_leg: bool = False,
        trailing_incomplete: bool = False,
    ) -> None:
        """Accumulate timeout, degraded-leg and trailing-loss flags; each stays true once set.

        `trailing_incomplete` marks a stream that dropped a slot-options or observations
        event (`enrichment._stream_slot_options`); its bundle is not stored. No flag reaches
        the wire."""
        self._timed_out = self._timed_out or timed_out
        self._degraded_leg = self._degraded_leg or degraded_leg
        self._trailing_incomplete = self._trailing_incomplete or trailing_incomplete

    @property
    def degraded_leg(self) -> bool:
        """Whether a fresh run noted a degraded leg. A cache hit reports False."""
        return self._degraded_leg

    async def observe(self, event: Event) -> None:
        """Take one event of the live stream; publish the answer once it is complete.

        A bundle request records events from `response` and publishes at `done`, leaving
        out place options, which a hit runs again. Any other request publishes its response
        at `response`.
        """
        if not self.active:
            return
        if not self._bundle:
            if isinstance(event, ResponseEvent):
                await self.store(event.response)
            return
        if isinstance(event, PlaceOptionsEvent):
            return
        if isinstance(event, ResponseEvent) or self._recorded:
            self._recorded.append(event)
        if isinstance(event, DoneEvent):
            await self._publish(
                lambda: to_bundle(self._recorded),
                lambda: is_bundle_storable(
                    self._recorded,
                    timed_out=self._timed_out,
                    degraded_leg=self._degraded_leg,
                    trailing_incomplete=self._trailing_incomplete,
                    refused_at_build=self._refused_at_build,
                ),
            )

    async def store(self, response: ResolveResponse) -> None:
        """Publish this request's response to any leases it owns.

        Only a request without trailing events stores a bare response. A streamed one
        publishes its bundle through `observe`, since its keys name a bundle entry.

        Followers always receive the answer, even when `is_storable` rejects it; that
        verdict controls storage, not delivery. Fresh runs use `note_health`; hits carry
        the leader's verdict because degraded-leg status is absent from response bytes. A
        hit is storable only when both its served verdict and this response's checks allow
        it. Published timeout status combines this run's and the leader's flags.
        """
        assert not self._bundle, "a streamed request publishes its bundle through observe()"
        await self._publish(
            lambda: to_entry(response),
            lambda: is_storable(
                response,
                timed_out=self._timed_out,
                degraded_leg=self._degraded_leg,
                refused_at_build=self._refused_at_build,
            ),
        )

    async def _publish(
        self,
        encode: Callable[[], tuple[bytes, int]],
        storable: Callable[[], bool],
    ) -> None:
        """Publish an entry to any leases this request owns.

        Cache errors are logged and never reach the pipeline. Both leases share one
        encoding and verdict. Entry lifetime comes from the generation stamp captured
        before lookup.
        """
        if not self.active:
            return
        assert self._cache is not None
        if self._layer1_lease is None and self._layer2_lease is None:
            # No lease means this request cannot publish a cache entry.
            return
        try:
            verdict = not self._fallbacks.reasons and self._served_storable and storable()
            timed_out = self._timed_out or self._served_timed_out
            entry, size = encode()
            for lease in (self._layer1_lease, self._layer2_lease):
                if lease is not None:
                    await self._cache.publish(
                        lease,
                        entry,
                        size,
                        storable=verdict,
                        timed_out=timed_out,
                        max_age_s=self._stamp.max_age_s,
                    )
            self._layer1_lease = None
            self._layer2_lease = None
        except Exception:
            logger.exception("response cache store failed; serving the fresh answer uncached")

    async def release(self) -> None:
        """Abandon unpublished leases, usually from `finally`, to wake all waiters.

        Safe after `store()`, which clears published leases.
        """
        if not self.active or self._cache is None:
            return
        if self._layer1_lease is not None:
            await self._cache.abandon(self._layer1_lease)
            self._layer1_lease = None
        if self._layer2_lease is not None:
            await self._cache.abandon(self._layer2_lease)
            self._layer2_lease = None

    def _restore(
        self,
        entry: bytes,
        *,
        query: str | None,
        extract_skipped: bool,
        llm_usage: dict[str, int] | None,
        step: str,
    ) -> ResolveResponse:
        elapsed = now_ms() - self._start_ms
        if self._bundle:
            self._replay = from_bundle(entry)
            return refresh_response(
                self._replay.response.response,
                query=query,
                extract_skipped=extract_skipped,
                start_ms=self._start_ms,
                llm_usage=llm_usage,
                timing_by_step={step: elapsed},
            )
        return from_entry(
            entry,
            query=query,
            extract_skipped=extract_skipped,
            start_ms=self._start_ms,
            llm_usage=llm_usage,
            timing_by_step={step: elapsed},
        )

    async def hit_events(
        self,
        response: ResolveResponse,
        *,
        start_ms: int,
        llm_usage: dict | None = None,
        place_options: ReplayedPlaceOptions | None = None,
    ) -> AsyncGenerator[Event]:
        """The events a cache hit sends for the `response` a lookup returned.

        The `response` and `done` events are fresh (trace id, timing, usage). A bundle hit
        sends the stored slot-options and observations events after `response`, then the
        place options a fresh run streams, launched by `place_options` and awaited under
        the place-options budget (`enrichment.stream_place_options`), and keeps the stored
        `done` event's warnings and late departures. Any other hit sends `response` and
        `done` alone. `place_options` is `None` when the request streams no place options.
        """
        yield ResponseEvent(response=response)
        if self._replay is not None:
            for event in self._replay.trailing:
                yield event
            if place_options is not None:
                async with aclosing(
                    stream_place_options(place_options(response), deadline=self._deadline)
                ) as streamed:
                    async for event in streamed:
                        yield event
        _, done_event = _terminal_events(
            response, start_ms=start_ms, llm_usage=llm_usage, timed_out=self._served_timed_out
        )
        if self._replay is not None:
            done_event = done_event.model_copy(
                update={
                    "warnings": self._replay.done.warnings,
                    "late_departures": self._replay.done.late_departures,
                }
            )
        yield done_event


class PartMemoScope:
    """One `dataset_resubmit` request's access to the part memo.

    Active when an enabled memo is given and `SEARCH_RESPONSE_CACHE_ENABLED` is on; an
    inactive scope looks up nothing and stores nothing. The generation stamp, the
    conversion catalogue revision, and the origin catalogue's build and UN System setting
    are read once, at construction, beside the build commit, coverage stamp and UTC day the
    caller passes, and key every identity entry the request looks up
    (`response_cache_keys.part_memo_key`). The stamp also sets each stored entry's
    lifetime. `hits` and `misses` count this request's lookups.
    """

    def __init__(
        self,
        memo: PartMemo | None,
        *,
        build_commit: str,
        today: str,
        coverage: CoverageSnapshot,
        generation: GenerationProvider = UNKNOWN_GENERATION,
    ) -> None:
        self._memo = (
            memo
            if memo is not None and memo.enabled and config.settings().response_cache_enabled
            else None
        )
        self._stamp = generation.stamp()
        conversion = current_conversion_catalogue()
        self._build_commit = build_commit
        self._today = today
        self._coverage_stamp = coverage.stamp
        self._conversion = conversion.revision if conversion is not None else None
        origin = current_catalogue()
        self._origin_catalogue = origin.build_id
        self._un_system_enabled = origin.un_system_enabled
        self._refused_at_build = origin.un_blocked_places
        self.hits = 0
        self.misses = 0

    @property
    def active(self) -> bool:
        return self._memo is not None

    def key(self, payload: ResubmitIdentityPayload, options: EngineOptions) -> str:
        """The memo key of `payload` resolved under `options`' four enrichment flags."""
        return part_memo_key(
            payload=payload,
            engine_flags={
                "include_option_coverage": options.include_option_coverage,
                "stream_slot_options": options.stream_slot_options,
                "include_observations": options.include_observations,
                "stream_observations": options.stream_observations,
            },
            build_commit=self._build_commit,
            graph_generation=self._stamp.key,
            coverage_stamp=self._coverage_stamp,
            conversion_catalogue=self._conversion,
            origin_catalogue=self._origin_catalogue,
            un_system_enabled=self._un_system_enabled,
            today=self._today,
        )

    def lookup(self, key: str) -> RegionResult | None:
        """The stored region under `key`, holding no closure, or `None`."""
        if self._memo is None:
            return None
        region = self._memo.get(self._stamp.key, key)
        if region is None:
            self.misses += 1
        else:
            self.hits += 1
        return region

    def store(self, key: str, region: RegionResult) -> None:
        """Store `region` without its closures when `is_region_storable` accepts it."""
        if self._memo is None or not is_region_storable(
            region, refused_at_build=self._refused_at_build
        ):
            return
        self._memo.put(self._stamp.key, key, without_relaunches(region))
