"""Resolve text to graph-confirmed entities through resolvekit and Data Commons.

`resolve()` preserves input order and caches by normalized text and one generation stamp captured
per call. Pending texts use the shared single-worker resolvekit executor, since resolvekit's
query cache and sqlite store are thread-affine. Query aliases run before resolvekit. A listed-name gate filters candidates. Parse can add
group results after a miss, and one retry strips a leading article and trailing punctuation.

Survivors and group members map to dcids. Misses use the description-arc ladder, which accepts
only nodes listing the normalized query and keeps all matches. One graph read confirms every hit
and group member. Results are `Resolved`, `Group`, `Ambiguous`, `NotHeld`, or `NotFound`. `NotFound`
includes up to three rejected candidate names as `near_matches`.

Graph reads use `graph_executor` or the loop's default executor and receive the caller's absolute
monotonic deadline. `GraphReadError` and `EntityResolutionError` propagate. Failed batches leave
the cache unchanged."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass

from cachetools import TLRUCache
from resolvekit import GroupNotFoundError, ResolutionContext, ResolutionStatus, Resolver
from resolvekit.core.model.result import CandidateSummary

from dfo_platform.executor import run_in_executor_with_context
from entity_resolution import pack
from entity_resolution.gate import filter_listed_candidates
from entity_resolution.graph import resolve_fallback
from entity_resolution.regions import identity_to_dcid
from entity_resolution.shared import SharedResolver
from entity_resolution.types import (
    ADMITTED_TYPES,
    Ambiguous,
    Entity,
    GraphLookup,
    Group,
    NotFound,
    NotHeld,
    Resolution,
    Resolved,
    entity_kind,
)
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from graph_generation.stamp import ttu_from_stamp_key

_CONFIRM_PROPERTIES = ("typeOf", "name")
_UNTYPED_GRAPH_TYPE = ""
_MAX_NEAR_MATCHES = 3
_ORG_ENTITY_TYPE = "org.organisation"

# Resolvekit's `known_groups()` omits entity types, so keep the two group types here.
_GROUP_TYPES: frozenset[str] = frozenset({"geo.organization", "geo.continental_union"})

# Pair each normalized text with the generation stamp captured for the whole call.
_CacheKey = tuple[str, str]

# Derive cache expiry from the generation stamp key. `Admission` uses the same rule.
_ttu = ttu_from_stamp_key


class EntityResolutionError(Exception):
    """Internal resolvekit failure for one text in a batch.

    Raised when resolvekit returns `ResolutionStatus.ERROR`."""


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


_LEADING_ARTICLE_RE = re.compile(r"^\s*the\s+", re.IGNORECASE)
_TRAILING_PUNCTUATION_RE = re.compile(r"[.,;:!?]+$")


def _without_article_and_punctuation(text: str) -> str:
    """Strip a leading `the` and trailing punctuation for a retry after a miss.

    Names such as `the Gambia` reach the primary pass intact. Return `text` unchanged when
    neither pattern matches."""
    stripped = _LEADING_ARTICLE_RE.sub("", text)
    stripped = _TRAILING_PUNCTUATION_RE.sub("", stripped)
    return stripped.strip()


def _load_query_aliases() -> dict[str, str]:
    """Load query aliases keyed by `_normalize_key`.

    Entries correct bundled pack name conflicts such as `Central Africa` and cover group phrases
    that `resolvekit.parse()` cannot recover, including `EU countries`, `EU member states`, and
    `DAC members`. Texts without an alias pass through unchanged."""
    return json.loads(pack.query_aliases_path().read_text(encoding="utf-8"))


_QUERY_ALIASES: dict[str, str] = _load_query_aliases()


@dataclass(frozen=True, slots=True)
class _GateOutcome:
    """Gate survivors, rejected-candidate suggestions, and optional raw group member IDs.

    `None` marks a non-group. An empty tuple records a group with no members. Build this on the
    resolver thread, where names and confidence scores are available."""

    survivors: tuple[CandidateSummary, ...]
    near_matches: tuple[str, ...]
    group_member_ids: tuple[str, ...] | None = None


def _candidate_name(candidate: CandidateSummary) -> str:
    return candidate.canonical_name or candidate.entity_id


def _near_matches(
    candidates: Sequence[CandidateSummary], survivors: Sequence[CandidateSummary]
) -> tuple[str, ...]:
    """Return up to `_MAX_NEAR_MATCHES` rejected canonical names, ordered by confidence then ID
    and deduplicated. Attach these suggestions to `NotFound`."""
    survivor_ids = {candidate.entity_id for candidate in survivors}
    rejected = sorted(
        (candidate for candidate in candidates if candidate.entity_id not in survivor_ids),
        key=lambda c: (-(c.confidence if c.confidence is not None else 0.0), c.entity_id),
    )
    names: list[str] = []
    seen: set[str] = set()
    for candidate in rejected:
        name = _candidate_name(candidate)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) == _MAX_NEAR_MATCHES:
            break
    return tuple(names)


def _parse_group_survivors(
    resolver: Resolver, text: str, context: ResolutionContext, group_types: frozenset[str]
) -> tuple[CandidateSummary, ...]:
    """Recover a membership-group span after full-text resolution and the listed-name gate miss.

    `parse()` bypasses query aliases and the gate, so apply both to each span and accept only
    spans whose type and top surviving candidate type are in `group_types`.

    Resolvekit excludes `geo.continent` and `geo.continental_union` spans. Aliases cover phrases
    such as `EU countries` and `EU member states`. `DAC members` parses as `geo.region` and fails
    the group-type check."""
    for entity in resolver.parse(text, context=context).entities:
        if entity.entity_type not in group_types:
            continue
        survivors = tuple(
            filter_listed_candidates(resolver, entity.surface, entity.resolution.candidates)
        )
        if survivors and survivors[0].entity_type in group_types:
            return survivors
    return ()


def _resolve_text_survivors(
    resolver: Resolver, text: str, context: ResolutionContext
) -> tuple[tuple[CandidateSummary, ...], tuple[CandidateSummary, ...]]:
    """Apply query aliases, resolve, gate candidates, and parse group spans after a miss.

    Return survivors and raw `resolve()` candidates. Near matches use the raw list when group
    survivors come from `parse()`."""
    query_text = _QUERY_ALIASES.get(_normalize_key(text), text)
    result = resolver.resolve(query_text, as_result=True, context=context, include_entity=True)
    if result.status == ResolutionStatus.ERROR:
        raise EntityResolutionError(f"resolvekit reported an internal error resolving {text!r}")
    survivors = tuple(filter_listed_candidates(resolver, query_text, result.candidates))
    if not survivors:
        survivors = _parse_group_survivors(resolver, text, context, _GROUP_TYPES)
    return survivors, tuple(result.candidates)


def _resolve_batch_job(resolver: Resolver, texts: Sequence[str]) -> list[_GateOutcome]:
    """Resolve and gate the batch on the shared resolver's single worker.

    Retry each miss once with a leading article and trailing punctuation stripped. The retry
    calls `resolve()`, not `parse()`, because `parse()` drops every continental-union span
    ("the EU"). For a group survivor, call `members_of` with its canonical name and omit
    `as_of` to avoid snapshot-group warnings. Catch `GroupNotFoundError`. That text follows ordinary hit assembly."""
    context = ResolutionContext(entity_types=ADMITTED_TYPES)
    outcomes: list[_GateOutcome] = []
    for text in texts:
        survivors, raw_candidates = _resolve_text_survivors(resolver, text, context)
        if not survivors:
            retry_text = _without_article_and_punctuation(text)
            if retry_text and retry_text != text:
                retry_survivors, _ = _resolve_text_survivors(resolver, retry_text, context)
                if retry_survivors:
                    survivors = retry_survivors
        group_member_ids: tuple[str, ...] | None = None
        if survivors and survivors[0].entity_type in _GROUP_TYPES:
            with contextlib.suppress(GroupNotFoundError):
                group_member_ids = tuple(resolver.members_of(_candidate_name(survivors[0])))
        outcomes.append(
            _GateOutcome(
                survivors=survivors,
                near_matches=_near_matches(raw_candidates, survivors),
                group_member_ids=group_member_ids,
            )
        )
    return outcomes


def _entity_from_confirmation(*, dcid: str, arcs: dict[str, tuple[str, ...]], name: str) -> Entity:
    type_values = arcs.get("typeOf", ())
    graph_type = type_values[0] if type_values else _UNTYPED_GRAPH_TYPE
    return Entity(dcid=dcid, name=name, kind=entity_kind(graph_type), graph_type=graph_type)


def _fallback_name(*, arcs: dict[str, tuple[str, ...]], text: str) -> str:
    names = arcs.get("name", ())
    return names[0] if names else text


class EntityResolver:
    """One shared resolver's own pipeline. `graph` and `shared` are both
    required: the pipeline reads through both on every non-empty,
    non-cached text. `graph_executor=None` uses the running loop's default
    executor for every graph call. `generation` defaults to
    `UNKNOWN_GENERATION`, so an uninjected instance keys and expires every
    entry under the "unknown" stamp (1h).
    """

    def __init__(
        self,
        *,
        graph: GraphLookup,
        shared: SharedResolver,
        graph_executor: Executor | None = None,
        cache_size: int = 4096,
        generation: GenerationProvider = UNKNOWN_GENERATION,
    ) -> None:
        self._graph = graph
        self._shared = shared
        self._graph_executor = graph_executor
        self._generation = generation
        # Generation refresh clears this cache on a background thread. Guard every cache access.
        self._cache_lock = threading.Lock()
        self._cache: TLRUCache[_CacheKey, Resolution] = TLRUCache(maxsize=cache_size, ttu=_ttu)
        self._hits = 0
        self._misses = 0

    def stats(self) -> Mapping[str, int]:
        with self._cache_lock:
            entries = len(self._cache)
        return {"hits": self._hits, "misses": self._misses, "entries": entries}

    def clear(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    async def resolve(
        self, texts: Sequence[str], *, deadline: float | None = None
    ) -> list[Resolution]:
        """Pass an absolute `time.monotonic()` deadline to every graph read.

        Propagate `GraphDeadlineExceededError` unchanged. `None` leaves reads unbounded. Capture
        one generation stamp before lookup so all texts in this call use the same generation."""
        stamp = self._generation.stamp()
        results: dict[int, Resolution] = {}
        keys: dict[int, _CacheKey] = {}
        pending_texts: dict[int, str] = {}

        for index, raw in enumerate(texts):
            stripped = raw.strip()
            if not stripped:
                results[index] = NotFound()
                continue
            key = (stamp.key, _normalize_key(stripped))
            with self._cache_lock:
                cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                results[index] = cached
                continue
            self._misses += 1
            keys[index] = key
            pending_texts[index] = stripped

        if pending_texts:
            await self._resolve_pending(results, keys, pending_texts, deadline=deadline)

        return [results[index] for index in range(len(texts))]

    async def _resolve_pending(
        self,
        results: dict[int, Resolution],
        keys: dict[int, _CacheKey],
        pending_texts: dict[int, str],
        *,
        deadline: float | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        pending_indices = list(pending_texts)
        ordered_texts = [pending_texts[index] for index in pending_indices]
        gate_outcomes = await loop.run_in_executor(
            self._shared.executor, _resolve_batch_job, self._shared.resolver, ordered_texts
        )

        hits: dict[int, list[tuple[CandidateSummary, str]]] = {}
        miss_texts: dict[int, str] = {}
        near_matches_by_index: dict[int, tuple[str, ...]] = {}
        group_member_dcids: dict[int, tuple[str, ...]] = {}
        for index, outcome in zip(pending_indices, gate_outcomes, strict=True):
            mapped = sorted(
                (
                    (candidate, dcid)
                    for candidate in outcome.survivors
                    if (dcid := identity_to_dcid(candidate.entity_id)) is not None
                ),
                key=lambda pair: (
                    -(pair[0].confidence if pair[0].confidence is not None else 0.0),
                    pair[1],
                ),
            )
            if mapped:
                hits[index] = mapped
            else:
                miss_texts[index] = pending_texts[index]
                near_matches_by_index[index] = outcome.near_matches
            if outcome.group_member_ids is not None:
                group_member_dcids[index] = tuple(
                    sorted(
                        {
                            dcid
                            for member_id in outcome.group_member_ids
                            if (dcid := identity_to_dcid(member_id)) is not None
                        }
                    )
                )

        fallback_dcids: dict[int, tuple[str, ...]] = {}
        if miss_texts:
            fallback = functools.partial(
                resolve_fallback,
                self._graph,
                normalize=self._shared.resolver.pack_normalizers["geo"].normalize,
                deadline=deadline,
            )
            fallback_dcids = await run_in_executor_with_context(
                self._graph_executor, fallback, miss_texts
            )

        all_dcids = sorted(
            {dcid for candidate_hits in hits.values() for _, dcid in candidate_hits}
            | {dcid for dcids in fallback_dcids.values() for dcid in dcids}
            | {dcid for dcids in group_member_dcids.values() for dcid in dcids}
        )
        confirmed: dict[str, dict[str, tuple[str, ...]] | None] = {}
        if all_dcids:
            confirm = functools.partial(
                self._graph.node_arcs_many, properties=_CONFIRM_PROPERTIES, deadline=deadline
            )
            confirmed = await run_in_executor_with_context(self._graph_executor, confirm, all_dcids)

        for index in pending_indices:
            # Group membership, including an empty tuple, can exist without a dcid for the group.
            # Use ordinary miss handling unless that candidate also maps.
            if index in group_member_dcids and index in hits:
                result = self._assemble_group(hits[index], group_member_dcids[index], confirmed)
            elif index in hits:
                result = self._assemble_hit(hits[index], confirmed)
            elif index in fallback_dcids:
                result = self._assemble_fallback(
                    fallback_dcids[index],
                    pending_texts[index],
                    confirmed,
                    near_matches_by_index[index],
                )
            else:
                result = NotFound(near_matches=near_matches_by_index[index])
            results[index] = result
            with self._cache_lock:
                self._cache[keys[index]] = result

    def _assemble_hit(
        self,
        candidate_hits: list[tuple[CandidateSummary, str]],
        confirmed: dict[str, dict[str, tuple[str, ...]] | None],
    ) -> Resolution:
        confirmed_hits = [
            (candidate, dcid)
            for candidate, dcid in candidate_hits
            if confirmed.get(dcid) is not None
        ]
        if not confirmed_hits:
            first_candidate, first_dcid = candidate_hits[0]
            return NotHeld(dcid=first_dcid, name=_candidate_name(first_candidate))
        entities = tuple(
            _entity_from_confirmation(
                dcid=dcid, arcs=confirmed[dcid] or {}, name=_candidate_name(candidate)
            )
            for candidate, dcid in confirmed_hits
        )
        if len(entities) == 1:
            return Resolved(entity=entities[0])
        return Ambiguous(candidates=entities)

    def _assemble_group(
        self,
        candidate_hits: list[tuple[CandidateSummary, str]],
        member_dcids: tuple[str, ...],
        confirmed: dict[str, dict[str, tuple[str, ...]] | None],
    ) -> Resolution:
        """`candidate_hits` starts with the winning group and its ordered survivors.
        Its ID maps directly to a dcid. `member_dcids` are translated and deduplicated.

        Prefer the confirmed group node for `entity`, then the first confirmed organisation
        survivor. Return `NotHeld` when no member is graph-confirmed."""
        winning_candidate, winning_dcid = candidate_hits[0]
        entity: Entity | None = None
        winning_arcs = confirmed.get(winning_dcid)
        if winning_arcs is not None:
            entity = _entity_from_confirmation(
                dcid=winning_dcid, arcs=winning_arcs, name=_candidate_name(winning_candidate)
            )
        else:
            for candidate, dcid in candidate_hits[1:]:
                if candidate.entity_type != _ORG_ENTITY_TYPE:
                    continue
                arcs = confirmed.get(dcid)
                if arcs is not None:
                    entity = _entity_from_confirmation(
                        dcid=dcid, arcs=arcs, name=_candidate_name(candidate)
                    )
                    break

        members = tuple(
            _entity_from_confirmation(
                dcid=dcid, arcs=arcs, name=_fallback_name(arcs=arcs, text=dcid)
            )
            for dcid in sorted(member_dcids)
            if (arcs := confirmed.get(dcid)) is not None
        )
        if not members:
            # Empty membership or zero graph-confirmed members is a data condition.
            # Return `NotHeld` even when the group entity is confirmed.
            return NotHeld(dcid=winning_dcid, name=_candidate_name(winning_candidate))
        return Group(members=members, entity=entity)

    def _assemble_fallback(
        self,
        dcids: tuple[str, ...],
        text: str,
        confirmed: dict[str, dict[str, tuple[str, ...]] | None],
        near_matches: tuple[str, ...],
    ) -> Resolution:
        """Return `Resolved`, `Ambiguous`, or `NotFound`.

        Sort fallback candidates by dcid. Each candidate is a graph node listing the input text."""
        entities = tuple(
            _entity_from_confirmation(
                dcid=dcid, arcs=arcs, name=_fallback_name(arcs=arcs, text=text)
            )
            for dcid in sorted(dcids)
            if (arcs := confirmed.get(dcid)) is not None
        )
        if not entities:
            return NotFound(near_matches=near_matches)
        if len(entities) == 1:
            return Resolved(entity=entities[0])
        return Ambiguous(candidates=entities)
