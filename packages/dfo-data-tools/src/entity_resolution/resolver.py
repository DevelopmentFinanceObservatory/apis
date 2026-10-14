"""Resolve text to graph-confirmed entities through resolvekit and Data Commons.

`resolve()` preserves input order and caches by normalized text, the requested kinds and code
system, and one generation stamp captured per call. Pending texts use the shared single-worker
resolvekit executor, since resolvekit's query cache and sqlite store are thread-affine; a job
whose deadline passed while it queued does not start. Query aliases run before resolvekit. Pack
rows that store the text as a name join resolvekit's candidates. A listed-name gate filters
candidates and records whether each lists the text as its name, an alias or a code. Parse can
add group results after a miss, and one retry strips a leading article and trailing punctuation.

A top survivor the group registry holds resolves as a `Group`: the registry confirms it and
its members, so no graph read does. Other survivors map to dcids. Misses use the
description-arc ladder, which accepts only nodes listing the normalized query and keeps all
matches. The `code` kind adds the codes whose label or synonym lists the text under the cache
key's fold. One graph read confirms every hit, group node and code. A group or region the
membership policy holds counts as held with or without a graph node, as on the Graph API's
node and members routes. Every candidate takes its kind by the rule the node route classes
by (`kinds.classify`), over the code index the resolver was built with and the membership
policy: a confirmed candidate by its `typeOf` values, and one the graph does not hold by no
types, falling back to its resolvekit kind (`_unconfirmed_kind`). A `kinds` filter then
keeps the candidates of the requested kinds; one left resolves, and none left returns
`NotFound` naming the removed candidates.
Results are `Resolved`, `Group`, `Ambiguous`, `NotHeld`, or `NotFound`. `NotFound` includes up
to three rejected candidates as `near_matches`, unconfirmed by the graph.

Graph reads use `graph_executor` or the loop's default executor and receive the caller's absolute
monotonic deadline. `GraphReadError` and `EntityResolutionError` propagate. Failed batches leave
the cache unchanged."""

from __future__ import annotations

import asyncio
import functools
import json
import re
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field

from cachetools import TLRUCache
from resolvekit import ResolutionContext, ResolutionStatus, Resolver
from resolvekit.core.model.result import CandidateSummary

from dc_graph.errors import GraphDeadlineExceededError
from dfo_platform.executor import run_in_executor_with_context
from entity_resolution import pack
from entity_resolution.gate import filter_listed_candidates
from entity_resolution.graph import normalize_or_none, resolve_fallback
from entity_resolution.kinds import classify
from entity_resolution.membership import MembershipPolicy, NotRecorded, UnknownMembership
from entity_resolution.regions import identity_to_dcid
from entity_resolution.shared import SharedResolver
from entity_resolution.types import (
    ADMITTED_TYPES,
    Ambiguous,
    CodeLookup,
    Entity,
    EntityKind,
    GraphLookup,
    Group,
    Kind,
    KindIndex,
    Matched,
    MatchedAs,
    NotFound,
    NotHeld,
    Reading,
    Resolution,
    Resolved,
    candidate_kind,
)
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from graph_generation.stamp import ttu_from_stamp_key

_CONFIRM_PROPERTIES = ("typeOf", "name")
_UNTYPED_GRAPH_TYPE = ""
_MAX_NEAR_MATCHES = 3
_ORG_ENTITY_TYPE = "org.organisation"
_CODE_KIND: EntityKind = "code"
_GROUP_KIND: EntityKind = "group"
_UNCLASSED_KIND = "other"

# The default projection holds countries alone, so every member a `Group` carries is one.
_MEMBER_KIND: EntityKind = "country"
_MEMBER_GRAPH_TYPE = "Country"

# Resolvekit's `known_groups()` omits entity types, so keep the two group types here.
_GROUP_TYPES: frozenset[str] = frozenset({"geo.organization", "geo.continental_union"})

# The generation stamp captured for the whole call, the normalized text, the requested kinds
# (`None` for no filter) and the code system.
_CacheKey = tuple[str, str, frozenset[EntityKind] | None, str | None]

# Derive cache expiry from the generation stamp key. `Admission` uses the same rule.
_ttu = ttu_from_stamp_key

# A confirmed candidate with the listed text it matched.
_Candidate = tuple[Entity, Matched | None]


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


@dataclass(frozen=True, slots=True)
class _QueryAlias:
    """The pack name a query text resolves as. `members` is true when the text names the
    members of the set that name resolves to, each as its own place."""

    name: str
    members: bool = False


def _load_query_aliases() -> dict[str, _QueryAlias]:
    """Load query aliases keyed by `_normalize_key`.

    Entries correct bundled pack name conflicts such as `Central Africa` and cover group phrases
    that `resolvekit.parse()` cannot recover, including `EU countries`, `EU member states`, and
    `DAC members`. Texts without an alias pass through unchanged."""
    rows = json.loads(pack.query_aliases_path().read_text(encoding="utf-8"))
    return {key: _QueryAlias(**row) for key, row in rows.items()}


_QUERY_ALIASES: dict[str, _QueryAlias] = _load_query_aliases()


@dataclass(frozen=True, slots=True)
class _GroupHit:
    """The registry group a text's top survivor names, with its members under the default
    projection today, named by resolvekit."""

    ref: str
    label: str
    members: tuple[Entity, ...]


@dataclass(frozen=True, slots=True)
class _GateOutcome:
    """Gate survivors, rejected-candidate suggestions, how each survivor listed the text (by
    entity id), the registry group the top survivor names, and the text's reading.

    Build this on the resolver thread, where names and confidence scores are available."""

    survivors: tuple[CandidateSummary, ...]
    near_matches: tuple[Entity, ...]
    matched: Mapping[str, Matched] = field(default_factory=dict)
    group: _GroupHit | None = None
    reading: Reading = "entity"


# Gate survivors, each with the listed text it matched.
_Listed = tuple[tuple[CandidateSummary, Matched], ...]


def _candidate_name(candidate: CandidateSummary) -> str:
    return candidate.canonical_name or candidate.entity_id


def _confirmed_name(
    candidate: CandidateSummary, *, dcid: str, arcs: dict[str, tuple[str, ...]]
) -> str:
    """The candidate's pack name, or its graph name for a code-only row, which the pack names by
    its dcid."""
    name = _candidate_name(candidate)
    return _fallback_name(arcs=arcs, text=name) if name == dcid else name


def _unconfirmed_kind(
    policy: MembershipPolicy,
    code_index: KindIndex | None,
    *,
    dcid: str,
    candidate: CandidateSummary,
) -> Kind:
    """The kind of a candidate the graph has not confirmed: the kind rule's over no graph
    types (a held group or region, a code, an aggregate party), else the kind of its
    resolvekit type."""
    kind = classify(dcid=dcid, types=(), code_index=code_index, membership=policy)
    return candidate_kind(candidate.entity_type) if kind == _UNCLASSED_KIND else kind


def _near_matches(
    candidates: Sequence[CandidateSummary],
    survivors: Sequence[CandidateSummary],
    policy: MembershipPolicy,
    code_index: KindIndex | None,
) -> tuple[Entity, ...]:
    """Return up to `_MAX_NEAR_MATCHES` rejected candidates, ordered by confidence then ID and
    deduplicated by dcid. Attach these suggestions to `NotFound`. The graph confirms none of
    them, so `graph_type` is empty; a candidate with no dcid is left out. Each takes its kind
    by `_unconfirmed_kind`."""
    survivor_ids = {candidate.entity_id for candidate in survivors}
    rejected = sorted(
        (candidate for candidate in candidates if candidate.entity_id not in survivor_ids),
        key=lambda c: (-(c.confidence if c.confidence is not None else 0.0), c.entity_id),
    )
    matches: dict[str, Entity] = {}
    for candidate in rejected:
        dcid = identity_to_dcid(candidate.entity_id)
        if dcid is None or dcid in matches:
            continue
        matches[dcid] = Entity(
            dcid=dcid,
            name=_candidate_name(candidate),
            kind=_unconfirmed_kind(policy, code_index, dcid=dcid, candidate=candidate),
            graph_type=_UNTYPED_GRAPH_TYPE,
        )
        if len(matches) == _MAX_NEAR_MATCHES:
            break
    return tuple(matches.values())


def _listed(text: str, survivors: Sequence[tuple[CandidateSummary, MatchedAs]]) -> _Listed:
    return tuple((candidate, Matched(text=text, matched_as=how)) for candidate, how in survivors)


def _parse_group_survivors(
    resolver: Resolver, text: str, context: ResolutionContext, group_types: frozenset[str]
) -> _Listed:
    """Recover a membership-group span after full-text resolution and the listed-name gate miss.

    `parse()` bypasses query aliases and the gate, so apply both to each span and accept only
    spans whose type and top surviving candidate type are in `group_types`. Each survivor
    matched the span's text.

    Resolvekit excludes `geo.continent` and `geo.continental_union` spans. Aliases cover phrases
    such as `EU countries` and `EU member states`. `DAC members` parses as `geo.region` and fails
    the group-type check."""
    for entity in resolver.parse(text, context=context).entities:
        if entity.entity_type not in group_types:
            continue
        survivors = filter_listed_candidates(resolver, entity.surface, entity.resolution.candidates)
        if survivors and survivors[0][0].entity_type in group_types:
            return _listed(entity.surface, survivors)
    return ()


def _with_alias_owners(
    candidates: Sequence[CandidateSummary], alias_index: pack.AliasIndex, text: str
) -> tuple[CandidateSummary, ...]:
    """Append each `dfo_graph` row that stores `text` as a name and is missing from `candidates`.

    Resolvekit generates no candidate for many stored aliases. Its org exact-name and acronym
    sources skip `alias` rows, so `OAPEC` misses its organisation: resolvekit offers only
    `groups/OPEC`, which the gate rejects. The BYOD build stores each alias with its
    punctuation, which the org query profile strips, so `C&A Foundation` misses in every
    source. Appended owners carry no confidence, so they rank after resolvekit's
    candidates, and they pass the listed-name gate like any other candidate."""
    present = {candidate.entity_id for candidate in candidates}
    owners = tuple(owner for owner in alias_index.owners_of(text) if owner.entity_id not in present)
    return (*candidates, *owners)


def _resolve_text_survivors(
    shared: SharedResolver, text: str, context: ResolutionContext
) -> tuple[_Listed, tuple[CandidateSummary, ...], Reading]:
    """Apply query aliases, resolve, add alias owners, gate candidates, and parse group spans
    after a miss.

    Return survivors with the text each matched, the candidates the gate saw, and the reading:
    `members` when the survivors came through an alias that declares its text names members.
    A query alias is itself the listed text, so its survivors match `text` as an alias. Near
    matches use the gate's candidates when group survivors come from `parse()`."""
    resolver = shared.resolver
    alias = _QUERY_ALIASES.get(_normalize_key(text))
    query_text = text if alias is None else alias.name
    result = resolver.resolve(query_text, as_result=True, context=context, include_entity=True)
    if result.status == ResolutionStatus.ERROR:
        raise EntityResolutionError(f"resolvekit reported an internal error resolving {text!r}")
    candidates = _with_alias_owners(result.candidates, shared.alias_index, query_text)
    survivors = filter_listed_candidates(resolver, query_text, candidates)
    if survivors and alias is not None:
        reading: Reading = "members" if alias.members else "entity"
        as_alias = _listed(text, [(candidate, "alias") for candidate, _ in survivors])
        return as_alias, candidates, reading
    if survivors:
        return _listed(text, survivors), candidates, "entity"
    return _parse_group_survivors(resolver, text, context, _GROUP_TYPES), candidates, "entity"


def _group_hit(shared: SharedResolver, candidate: CandidateSummary) -> _GroupHit | None:
    """The registry group `candidate` names, or `None` when the registry holds no such group."""
    try:
        membership = shared.groups.by_entity_id(candidate.entity_id)
    except UnknownMembership:
        return None
    members = membership.members()
    member_dcids = () if isinstance(members, NotRecorded) else members
    return _GroupHit(
        ref=membership.ref,
        label=membership.label,
        members=tuple(
            Entity(
                dcid=dcid,
                name=membership.member_labels.get(dcid, dcid),
                kind=_MEMBER_KIND,
                graph_type=_MEMBER_GRAPH_TYPE,
            )
            for dcid in member_dcids
        ),
    )


def _resolve_batch_job(
    shared: SharedResolver, code_index: KindIndex | None, texts: Sequence[str]
) -> list[_GateOutcome]:
    """Resolve and gate the batch on the shared resolver's single worker.

    Retry each miss once with a leading article and trailing punctuation stripped. The retry
    calls `resolve()`, not `parse()`, because `parse()` drops every continental-union span
    ("the EU"). A top survivor the group registry holds carries the registry's group; any
    other text follows ordinary hit assembly."""
    context = ResolutionContext(entity_types=ADMITTED_TYPES)
    outcomes: list[_GateOutcome] = []
    for text in texts:
        survivors, raw_candidates, reading = _resolve_text_survivors(shared, text, context)
        if not survivors:
            retry_text = _without_article_and_punctuation(text)
            if retry_text and retry_text != text:
                survivors, _, reading = _resolve_text_survivors(shared, retry_text, context)
        candidates = tuple(candidate for candidate, _ in survivors)
        outcomes.append(
            _GateOutcome(
                survivors=candidates,
                near_matches=_near_matches(raw_candidates, candidates, shared.policy, code_index),
                matched={candidate.entity_id: matched for candidate, matched in survivors},
                group=_group_hit(shared, candidates[0]) if candidates else None,
                reading=reading,
            )
        )
    return outcomes


def _unless_expired[*Ts, R](deadline: float | None, fn: Callable[[*Ts], R], /, *args: *Ts) -> R:
    """Run `fn(*args)` unless the absolute monotonic `deadline` passed while the job queued.

    Raises:
        GraphDeadlineExceededError: the deadline passed before the job started.
    """
    if deadline is not None and time.monotonic() >= deadline:
        raise GraphDeadlineExceededError("the request's deadline passed while its resolve queued")
    return fn(*args)


def _group_node_dcids_job(shared: SharedResolver) -> frozenset[str]:
    """The dcids `Group.entity` can hold, for every group resolvekit knows: the group's own
    id as a dcid, and each organisation that survives the gate beside it
    (`EntityResolver._assemble_group`). A region the membership policy holds is left out,
    since the policy answers its members ("Africa (DAC)" resolves to `africa`). Runs on the
    shared resolver's single worker."""
    context = ResolutionContext(entity_types=ADMITTED_TYPES)
    dcids: set[str] = set()
    for name in shared.resolver.known_groups():
        listed, _, _ = _resolve_text_survivors(shared, name, context)
        survivors = [candidate for candidate, _ in listed]
        if not survivors or survivors[0].entity_type not in _GROUP_TYPES:
            continue
        nodes = [survivors[0]] + [c for c in survivors[1:] if c.entity_type == _ORG_ENTITY_TYPE]
        dcids.update(
            dcid for node in nodes if (dcid := identity_to_dcid(node.entity_id)) is not None
        )
    return frozenset(dcid for dcid in dcids if not shared.policy.holds_region(dcid))


def _entity_from_confirmation(
    policy: MembershipPolicy,
    code_index: KindIndex | None,
    *,
    dcid: str,
    arcs: dict[str, tuple[str, ...]],
    name: str,
) -> Entity:
    """The confirmed node `dcid`, classed by the kind rule over all its types
    (`kinds.classify`). A group takes the policy's label."""
    type_values = arcs.get("typeOf", ())
    graph_type = type_values[0] if type_values else _UNTYPED_GRAPH_TYPE
    kind = classify(dcid=dcid, types=type_values, code_index=code_index, membership=policy)
    if kind == _GROUP_KIND:
        name = policy.get(dcid).label
    return Entity(dcid=dcid, name=name, kind=kind, graph_type=graph_type)


def _fallback_name(*, arcs: dict[str, tuple[str, ...]], text: str) -> str:
    names = arcs.get("name", ())
    return names[0] if names else text


def _admits(kinds: frozenset[EntityKind] | None, entity: Entity) -> bool:
    return kinds is None or entity.kind in kinds


def _choose(
    candidates: Sequence[_Candidate],
    *,
    kinds: frozenset[EntityKind] | None,
    near_matches: tuple[Entity, ...],
    reading: Reading,
) -> Resolution:
    """Keep the candidates of the requested kinds, in order. One left resolves; several are
    ambiguous; none left is `NotFound`, which names the candidates the filter removed."""
    admitted = [pair for pair in candidates if _admits(kinds, pair[0])]
    if not admitted:
        return NotFound(near_matches=near_matches, removed=tuple(e for e, _ in candidates))
    if len(admitted) == 1:
        ((entity, matched),) = admitted
        return Resolved(entity=entity, matched=matched, reading=reading)
    return Ambiguous(candidates=tuple(entity for entity, _ in admitted))


class EntityResolver:
    """One shared resolver's own pipeline. `graph` and `shared` are both
    required: the pipeline reads through both on every non-empty,
    non-cached text. `graph_executor=None` uses the running loop's default
    executor for every graph call. `generation` defaults to
    `UNKNOWN_GENERATION`, so an uninjected instance keys and expires every
    entry under the "unknown" stamp (1h). `codes` serves the `code` kind; a
    resolver without it refuses a request for that kind. `code_index` gives the kind rule
    its code systems, codes and parties (`kinds.classify`); pass the one the Graph API's node
    route classes by, so both give a node one kind. Without it no node is one of those.
    """

    def __init__(
        self,
        *,
        graph: GraphLookup,
        shared: SharedResolver,
        graph_executor: Executor | None = None,
        cache_size: int = 4096,
        generation: GenerationProvider = UNKNOWN_GENERATION,
        codes: CodeLookup | None = None,
        code_index: KindIndex | None = None,
    ) -> None:
        self._graph = graph
        self._shared = shared
        self._graph_executor = graph_executor
        self._generation = generation
        self._codes = codes
        self._code_index = code_index
        # Generation refresh clears this cache on a background thread. Guard every cache access.
        self._cache_lock = threading.Lock()
        self._cache: TLRUCache[_CacheKey, Resolution] = TLRUCache(maxsize=cache_size, ttu=_ttu)
        self._hits = 0
        self._misses = 0
        self._group_node_dcids: frozenset[str] | None = None

    @property
    def membership(self) -> MembershipPolicy:
        """The shared membership policy, which answers a group's or a region's members."""
        return self._shared.policy

    def is_exact_registered_name(self, text: str) -> bool:
        """Whether ``text`` is an exact name held by the membership policy, the group registry
        or the graph pack. The registry still names a group the policy answers by a region
        ("Africa (DAC)" by `africa`, `membership.build_policy`).

        This is deliberately a narrow read-only registry check for callers that recover a
        source span. It does not parse or resolve a partial surface.
        """
        normalized = _normalize_key(text)
        memberships = (*self.membership, *self._shared.groups)
        if any(_normalize_key(membership.label) == normalized for membership in memberships):
            return True
        return bool(self._shared.alias_index.owners_of(text))

    def stats(self) -> Mapping[str, int]:
        with self._cache_lock:
            entries = len(self._cache)
        return {"hits": self._hits, "misses": self._misses, "entries": entries}

    def clear(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    async def group_node_dcids(self) -> frozenset[str]:
        """The dcids that are the node of a group the resolver knows, such as
        `EuropeanUnion`: every dcid a `Group.entity` can hold. The packs fix the set, so it is
        computed once and no graph is read."""
        if self._group_node_dcids is None:
            self._group_node_dcids = await asyncio.get_running_loop().run_in_executor(
                self._shared.executor, _group_node_dcids_job, self._shared
            )
        return self._group_node_dcids

    async def resolve(
        self,
        texts: Sequence[str],
        *,
        kinds: Collection[EntityKind] | None = None,
        code_system: str | None = None,
        deadline: float | None = None,
    ) -> list[Resolution]:
        """Resolve each text, keeping only candidates of `kinds` (every kind when `None`).

        `code_system` narrows the `code` kind to one system's graph id. Pass an absolute
        `time.monotonic()` deadline: the resolvekit job does not start once it has passed, and
        every graph read receives it. Propagate `GraphDeadlineExceededError` unchanged. `None`
        leaves the request unbounded. Capture one generation stamp before lookup so all texts in
        this call use the same generation.

        Raises:
            ValueError: `code_system` without the `code` kind, or the `code` kind on a resolver
                built without `codes`.
        """
        kind_filter = None if kinds is None else frozenset(kinds)
        wants_codes = kind_filter is not None and _CODE_KIND in kind_filter
        if code_system is not None and not wants_codes:
            raise ValueError("code_system narrows the code kind, which kinds does not include")
        if wants_codes and self._codes is None:
            raise ValueError("the code kind needs a code lookup, and this resolver has none")
        stamp = self._generation.stamp()
        results: dict[int, Resolution] = {}
        keys: dict[int, _CacheKey] = {}
        pending_texts: dict[int, str] = {}

        for index, raw in enumerate(texts):
            stripped = raw.strip()
            if not stripped:
                results[index] = NotFound()
                continue
            key = (stamp.key, _normalize_key(stripped), kind_filter, code_system)
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
            await self._resolve_pending(
                results,
                keys,
                pending_texts,
                kinds=kind_filter,
                code_system=code_system,
                deadline=deadline,
            )

        return [results[index] for index in range(len(texts))]

    def _code_candidates(
        self, key: _CacheKey, *, kinds: frozenset[EntityKind] | None, code_system: str | None
    ) -> list[_Candidate]:
        """The codes listing the folded text of the cache `key` when `kinds` asks for them,
        before graph confirmation. Each matched the code's own label or synonym, so two texts
        sharing a key share one answer."""
        if self._codes is None or kinds is None or _CODE_KIND not in kinds:
            return []
        _, folded, _, _ = key
        return [
            (hit.entity, Matched(text=hit.text, matched_as=hit.matched_as))
            for hit in self._codes.codes_by_text(folded, code_system=code_system)
        ]

    async def _resolve_pending(
        self,
        results: dict[int, Resolution],
        keys: dict[int, _CacheKey],
        pending_texts: dict[int, str],
        *,
        kinds: frozenset[EntityKind] | None,
        code_system: str | None,
        deadline: float | None,
    ) -> None:
        code_hits = {
            index: self._code_candidates(keys[index], kinds=kinds, code_system=code_system)
            for index in pending_texts
        }
        outcomes: dict[int, _GateOutcome] = {}
        # A request for codes alone runs no resolvekit job and no fallback ladder.
        if kinds is None or kinds - {_CODE_KIND}:
            pending_indices = list(pending_texts)
            ordered_texts = [pending_texts[index] for index in pending_indices]
            gate_outcomes = await asyncio.get_running_loop().run_in_executor(
                self._shared.executor,
                _unless_expired,
                deadline,
                _resolve_batch_job,
                self._shared,
                self._code_index,
                ordered_texts,
            )
            outcomes = dict(zip(pending_indices, gate_outcomes, strict=True))

        hits: dict[int, list[tuple[CandidateSummary, str]]] = {}
        miss_texts: dict[int, str] = {}
        for index, outcome in outcomes.items():
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

        fallback_dcids: dict[int, tuple[str, ...]] = {}
        if miss_texts:
            fallback = functools.partial(
                resolve_fallback,
                self._graph,
                normalize=self._normalize,
                deadline=deadline,
            )
            fallback_dcids = await run_in_executor_with_context(
                self._graph_executor, fallback, miss_texts
            )

        all_dcids = sorted(
            {dcid for candidate_hits in hits.values() for _, dcid in candidate_hits}
            | {dcid for dcids in fallback_dcids.values() for dcid in dcids}
            | {entity.dcid for candidates in code_hits.values() for entity, _ in candidates}
        )
        confirmed: dict[str, dict[str, tuple[str, ...]] | None] = {}
        if all_dcids:
            confirm = functools.partial(
                self._graph.node_arcs_many, properties=_CONFIRM_PROPERTIES, deadline=deadline
            )
            confirmed = await run_in_executor_with_context(self._graph_executor, confirm, all_dcids)

        for index in pending_texts:
            # A code counts only once the graph confirms it, like any other candidate.
            codes = [pair for pair in code_hits[index] if confirmed.get(pair[0].dcid) is not None]
            outcome = outcomes.get(index)
            group_admitted = kinds is None or _GROUP_KIND in kinds
            if outcome is None:
                result = _choose(codes, kinds=kinds, near_matches=(), reading="entity")
            elif outcome.group is not None and index in hits and group_admitted:
                result = self._assemble_group(outcome.group, hits[index], outcome, confirmed, codes)
            elif index in hits:
                result = self._assemble_hit(hits[index], outcome, confirmed, kinds, codes)
            elif index in fallback_dcids:
                result = self._assemble_fallback(
                    fallback_dcids[index],
                    pending_texts[index],
                    confirmed,
                    outcome.near_matches,
                    kinds,
                    codes,
                )
            else:
                result = _choose(
                    codes,
                    kinds=kinds,
                    near_matches=outcome.near_matches,
                    reading="entity",
                )
            self._store(results, keys, index, result)

    def _store(
        self,
        results: dict[int, Resolution],
        keys: dict[int, _CacheKey],
        index: int,
        result: Resolution,
    ) -> None:
        results[index] = result
        with self._cache_lock:
            self._cache[keys[index]] = result

    def _confirmed(self, *, dcid: str, arcs: dict[str, tuple[str, ...]], name: str) -> Entity:
        return _entity_from_confirmation(
            self._shared.policy, self._code_index, dcid=dcid, arcs=arcs, name=name
        )

    def _normalize(self, text: str) -> str:
        return self._shared.resolver.pack_normalizers["geo"].normalize(text)

    def _assemble_hit(
        self,
        candidate_hits: list[tuple[CandidateSummary, str]],
        outcome: _GateOutcome,
        confirmed: dict[str, dict[str, tuple[str, ...]] | None],
        kinds: frozenset[EntityKind] | None,
        codes: list[_Candidate],
    ) -> Resolution:
        """Choose among the confirmed hits and codes by `kinds`. A group or region the
        membership policy holds counts as a candidate classed over no graph types when the
        graph holds no node for it, at any rank, so a filter that removes it says so.

        A hit the graph does not hold takes its kind by `_unconfirmed_kind`.
        With no confirmed candidate of a requested kind, the first unheld hit of one is
        `NotHeld`; with none of either, `NotFound` names every hit and code the filter removed,
        in rank order."""
        candidates: list[_Candidate] = []
        unheld: list[Entity] = []
        ranked: list[Entity] = []
        for candidate, dcid in candidate_hits:
            arcs = confirmed.get(dcid)
            if arcs is not None:
                entity = self._confirmed(
                    dcid=dcid, arcs=arcs, name=_confirmed_name(candidate, dcid=dcid, arcs=arcs)
                )
            elif dcid in self._shared.policy:
                entity = self._confirmed(dcid=dcid, arcs={}, name=_candidate_name(candidate))
            else:
                entity = Entity(
                    dcid=dcid,
                    name=_candidate_name(candidate),
                    kind=_unconfirmed_kind(
                        self._shared.policy, self._code_index, dcid=dcid, candidate=candidate
                    ),
                    graph_type=_UNTYPED_GRAPH_TYPE,
                )
                unheld.append(entity)
                ranked.append(entity)
                continue
            ranked.append(entity)
            candidates.append((entity, outcome.matched.get(candidate.entity_id)))
        candidates.extend(codes)
        if any(_admits(kinds, entity) for entity, _ in candidates):
            return _choose(candidates, kinds=kinds, near_matches=(), reading=outcome.reading)
        if first_unheld := next((e for e in unheld if _admits(kinds, e)), None):
            return NotHeld(dcid=first_unheld.dcid, name=first_unheld.name)
        return NotFound(removed=(*ranked, *(entity for entity, _ in codes)))

    def _assemble_group(
        self,
        group: _GroupHit,
        candidate_hits: list[tuple[CandidateSummary, str]],
        outcome: _GateOutcome,
        confirmed: dict[str, dict[str, tuple[str, ...]] | None],
        codes: list[_Candidate],
    ) -> Resolution:
        """`candidate_hits` holds the group and the survivors beside it. The registry confirms
        the group and its members, so the result stands whether or not the graph holds them.

        Prefer the confirmed group node for `entity`, then the first confirmed organisation
        survivor. A code that also lists the text makes the result ambiguous between the group
        and the code."""
        matched = next(
            (
                outcome.matched.get(candidate.entity_id)
                for candidate, dcid in candidate_hits
                if dcid == group.ref
            ),
            None,
        )
        entity: Entity | None = None
        group_arcs = confirmed.get(group.ref)
        if group_arcs is not None:
            entity = self._confirmed(dcid=group.ref, arcs=group_arcs, name=group.label)
        else:
            for candidate, dcid in candidate_hits:
                if candidate.entity_type != _ORG_ENTITY_TYPE:
                    continue
                arcs = confirmed.get(dcid)
                if arcs is not None:
                    entity = self._confirmed(
                        dcid=dcid, arcs=arcs, name=_confirmed_name(candidate, dcid=dcid, arcs=arcs)
                    )
                    break
        if codes:
            node = Entity(
                dcid=group.ref,
                name=group.label,
                kind=_GROUP_KIND,
                graph_type=_UNTYPED_GRAPH_TYPE if entity is None else entity.graph_type,
            )
            return _choose(
                [(node, matched), *codes], kinds=None, near_matches=(), reading=outcome.reading
            )
        return Group(
            members=group.members,
            entity=entity,
            dcid=group.ref,
            label=group.label,
            matched=matched,
            reading=outcome.reading,
        )

    def _assemble_fallback(
        self,
        dcids: tuple[str, ...],
        text: str,
        confirmed: dict[str, dict[str, tuple[str, ...]] | None],
        near_matches: tuple[Entity, ...],
        kinds: frozenset[EntityKind] | None,
        codes: list[_Candidate],
    ) -> Resolution:
        """Choose among the confirmed fallback candidates and codes by `kinds`.

        Sort fallback candidates by dcid. Each is a graph node listing the input text, as its
        `name` or else as an alternate name (`alias`)."""
        normalized_text = normalize_or_none(self._normalize, text)
        candidates: list[_Candidate] = []
        for dcid in sorted(dcids):
            arcs = confirmed.get(dcid)
            if arcs is None:
                continue
            names = {normalize_or_none(self._normalize, name) for name in arcs.get("name", ())}
            matched_as: MatchedAs = (
                "name" if normalized_text is not None and normalized_text in names else "alias"
            )
            entity = self._confirmed(
                dcid=dcid, arcs=arcs, name=_fallback_name(arcs=arcs, text=text)
            )
            candidates.append((entity, Matched(text=text, matched_as=matched_as)))
        candidates.extend(codes)
        return _choose(candidates, kinds=kinds, near_matches=near_matches, reading="entity")
