"""Serve `GET /api/graph/schema/search`, the graph nodes a search box query reaches.

The legs run at once, each until `SCHEMA_SEARCH_LEG_SECONDS` from the request's start:

- `dcid`: a query in dcid syntax (`graph_api.routing.DcidConvertor.regex`) is read as a whole
  dcid from the node view's own readers, as `GET /api/graph/schema/node/{dcid}` reads it: a UN
  id from the UN deployment, any other from the mixer and, when the mixer does not hold it,
  from the UN deployment. The UN deployment is read only with `un=1` and the UN origin served.
  A deployment holds the node by the node view's rule: its `name` and `typeOf` read answers,
  or else its `->*` read, capped at `CENTRE_MAX_BYTES`, leaves an arc. A held node is one hit.
- `names`: the shared entity resolver as `POST /api/graph/resolve` runs it
  (`graph_api.doors.resolve.run_resolve`), without near matches. A resolved entity, or an
  ambiguous result's candidates, are hits. A query past `RESOLVE_TEXT_MAX_LENGTH` skips it.
  Its reads run on the shared DCP-read executor and resolvekit's one worker, which the
  resolve and node routes use too, so at most `SCHEMA_SEARCH_NAME_LEGS` names legs run at
  once per process; another waits for its turn within the leg's cap.
- `dfo`, `base` and `un_system`: one indicator `/v2/resolve` each: `custom_only` on the mixer,
  with `base=1` `base_only` on the mixer, and with `un=1` `custom_only` on the UN deployment
  where the bound catalogue serves the UN origin. Each keeps its first `INDICATOR_CANDIDATES`
  candidates in the resolver's order that are typed `StatisticalVariable` or name no type
  (`origin_catalogue.graph.parse_indicator_candidates`), skipping a candidate that fails the
  parser's checks, and each fails on its own.

Hits come in that order, each ref once, the first kept. Two reads of `name` and `typeOf`, one
batch on the mixer and one on the UN deployment where the request reads it, give each name and
indicator hit its label, types and `declared_by` (`graph_api.schema_document.declared_by` over
its `typeOf` targets' Provenances), each finishing by `SCHEMA_SEARCH_TIMEOUT_SECONDS` from the
start. An indicator hit a read that answered does not find is dropped, as is one the resolver
named no type for whose graph types lack `StatisticalVariable`. A read that fails names
`labels` in `unavailable`, and its hits stay with the label the resolver gave, none for an
indicator, the types they had (none for a name hit, `StatisticalVariable` for an indicator
typed so) and a null `declared_by`.

A leg that raises or runs out of time is named in `unavailable`, and the rest still answer
200. When every leg run fails the route answers retryable 503 `service_unavailable`, and when
every failure was a graph read answered HTTP 429, it answers as the schema routes answer a
rate-limited read, with `cause: {"code": "upstream_rate_limited"}` and the longest
`Retry-After` the reads named (`SchemaSearchRateLimited`). The dcid, indicator and label reads
run on an executor of `SCHEMA_SEARCH_WORKERS` workers of their own that refuses a read when
every worker is busy, which fails that leg. Every deadline counts from the request's start,
before its generation refresh.

An answer with no unavailable source is kept in an LRU of `SCHEMA_SEARCH_CACHE_SIZE` entries
keyed by the mixer's generation stamp, the UN deployment's where the request reads it, the
bound graph release, the origin catalogue's build, whether it serves the UN origin, the
stripped query and both switches, expiring with the mixer's stamp. The lifespan registers it as
`graph_api.schema_search`, so a mixer generation change clears it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, NamedTuple, Protocol

from cachetools import TLRUCache
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import StringConstraints

from dc_graph import ArcTarget, display_name
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.problem import Cause
from graph_api.deps import (
    DcpExecutorSaturated,
    DcpReadExecutor,
    GraphDeps,
    GraphDepsDep,
    GraphDepsNotSet,
    run_dcp_read,
)
from graph_api.doors.node import published_membership
from graph_api.doors.resolve import run_resolve
from graph_api.doors.schema import GRAPH_RATE_LIMITED_RETRY_AFTER_SECONDS
from graph_api.node_view import CENTRE_MAX_BYTES, NodeViewReader
from graph_api.resolve_models import RESOLVE_TEXT_MAX_LENGTH, ResolveRequest
from graph_api.routing import DcidConvertor
from graph_api.schema_document import declared_by, resolved_entry
from graph_api.schema_models import (
    SEARCH_QUERY_MAX_LENGTH,
    DeclaredBy,
    SchemaSearch,
    SchemaSearchHit,
    SearchSource,
)
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from graph_generation.stamp import ttu_from_stamp_key
from graph_release.context import served_origin_key
from materialise.graph import GraphReadError
from origin_catalogue import Catalogue, current_catalogue
from origin_catalogue.graph import IndicatorCandidate, is_un_dcid, parse_indicator_candidates

logger = logging.getLogger(__name__)

SCHEMA_SEARCH_LEG_SECONDS = 6.0
"""How long each leg runs from the request's start. The website's proxy sets no timeout of its
own, so the caller's timeout is the search box's; it should give up no sooner than 10 s."""

SCHEMA_SEARCH_TIMEOUT_SECONDS = 8.0
"""When the label reads of the hits must finish, from the request's start. The 2 s past the
legs' cap is the reads' own budget."""

SCHEMA_SEARCH_WORKERS = 16
"""The workers the legs' graph reads run on, per instance. One request holds at most three at
once (the dcid read and two indicator resolves), then two for the label reads, so five searches
with slow upstreams run at once before a sixth finds its reads refused."""

SCHEMA_SEARCH_CACHE_SIZE = 512
"""The most search answers the cache keeps."""

SCHEMA_SEARCH_NAME_LEGS = 1
"""The most names legs that run at once per process. Each holds at most one of the shared
DCP-read executor's `GRAPH_NODE_POOL_SIZE` workers (4) at a time and queues on resolvekit's one
worker, so the resolve and node routes keep the rest."""

INDICATOR_CANDIDATES = 10
"""The candidates each indicator leg keeps, in the resolver's order."""

_STATVAR = "StatisticalVariable"
_HIT_PROPERTIES = ("name", "typeOf")
_TYPE_OF = "typeOf"
_RATE_LIMITED = 429
_DCID = re.compile(DcidConvertor.regex)

# The failures a leg meets in normal running, logged without a traceback. Anything else is a
# fault in this code or a malformed answer, logged with one.
_EXPECTED_FAILURES: tuple[type[Exception], ...] = (
    TimeoutError,
    DcpExecutorSaturated,
    GraphReadError,
    # A resolve answer whose shape the parser refuses, logged on every request it recurs on.
    ValueError,
)

type SearchQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=SEARCH_QUERY_MAX_LENGTH),
]


class IndicatorResolver(Protocol):
    """The indicator `/v2/resolve` `origin_catalogue.graph.OriginGraphClient` posts."""

    def resolve_indicators(
        self, query: str, *, target: str, un: bool = False, deadline: float | None = None
    ) -> dict: ...


class SearchKey(NamedTuple):
    """The mixer's generation stamp key, the UN deployment's where the request reads it, the
    bound graph release, whose code index the entity resolver admits by, the origin catalogue's
    build, whether it serves the UN origin, the stripped query and the two switches.
    `ttu_from_stamp_key` reads the first member."""

    stamp: str
    un_stamp: str | None
    release: str
    origin_catalogue: str
    un_system_enabled: bool
    query: str
    base: bool
    un: bool


class SchemaSearchCache:
    """At most `maxsize` complete search answers, each expiring with its generation stamp. One
    lock covers every operation, since the generation provider calls `stats` and `clear` on its
    refresh thread."""

    def __init__(
        self,
        *,
        generation: GenerationProvider = UNKNOWN_GENERATION,
        maxsize: int = SCHEMA_SEARCH_CACHE_SIZE,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._entries: TLRUCache[SearchKey, SchemaSearch] = TLRUCache(
            maxsize=maxsize, ttu=ttu_from_stamp_key, timer=timer
        )
        self._generation = generation
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, key: SearchKey) -> SchemaSearch | None:
        with self._lock:
            found = self._entries.get(key)
            if found is None:
                self._misses += 1
            else:
                self._hits += 1
            return found

    def put(self, key: SearchKey, answer: SchemaSearch) -> None:
        """Store `answer`, unless the generation's stamp key is no longer `key.stamp`."""
        if self._generation.stamp().key != key.stamp:
            return
        with self._lock:
            self._entries[key] = answer

    def stats(self) -> Mapping[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "entries": len(self._entries)}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


@dataclass(frozen=True, slots=True)
class SchemaSearchDeps:
    """The search route's state: the indicator resolver, `None` on a surface with none, which
    reports its indicator legs unavailable; the executor its graph reads run on; its answer
    cache; the UN deployment's generation, which keys the answers that read it; and the
    semaphore that admits `SCHEMA_SEARCH_NAME_LEGS` names legs at once."""

    indicators: IndicatorResolver | None
    executor: DcpReadExecutor
    cache: SchemaSearchCache
    un_generation: GenerationProvider
    names: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(SCHEMA_SEARCH_NAME_LEGS)
    )


def new_search_executor() -> DcpReadExecutor:
    return DcpReadExecutor(
        max_workers=SCHEMA_SEARCH_WORKERS, thread_name_prefix="graph-schema-search"
    )


class SchemaSearchUnavailable(Exception):
    """Every leg the search ran failed or ran out of time."""

    def __init__(self, legs: Sequence[SearchSource]) -> None:
        super().__init__(f"no search leg answered ({', '.join(legs)}); retry")


class SchemaSearchRateLimited(Exception):
    """Every leg the search ran failed on a graph read answered HTTP 429. `retry_after` is the
    longest `Retry-After` those answers named, `None` where none named one."""

    def __init__(self, retry_after: float | None) -> None:
        super().__init__("the graph is limiting the rate of reads; retry")
        self.retry_after = retry_after


class _LegUnavailable(Exception):
    """A leg the surface cannot run, such as an indicator leg with no indicator resolver."""


@dataclass(frozen=True, slots=True)
class _Facts:
    label: str | None
    types: list[str]
    declared_by: DeclaredBy | None


@dataclass(frozen=True, slots=True)
class _Readers:
    """The node view's mixer reader and, where the request reads the UN deployment, its own."""

    mixer: NodeViewReader
    un: NodeViewReader | None


type _Entry = Mapping[str, Sequence[ArcTarget]]


def _facts(dcid: str, entry: _Entry, catalogue: Catalogue) -> _Facts:
    """`dcid`'s label, types and declaring origin from its `name` and `typeOf` targets."""
    type_targets = entry.get(_TYPE_OF, ())
    provenances = {p for target in type_targets for p in target.provenances}
    return _Facts(
        label=display_name((target.value for target in entry.get("name", ())), dcid),
        types=sorted({target.value for target in type_targets}),
        declared_by=declared_by(provenances, catalogue=catalogue) if provenances else None,
    )


def _read_facts(
    reader: NodeViewReader, dcids: Sequence[str], *, catalogue: Catalogue, deadline: float
) -> dict[str, _Facts]:
    """The facts of each of `dcids` `reader` holds, every empty answer read again alone."""
    if not dcids:
        return {}
    entries = reader.node_arc_targets_many(
        dcids, properties=_HIT_PROPERTIES, individual_probe_budget=None, deadline=deadline
    )
    return {
        dcid: _facts(dcid, entry, catalogue) for dcid, entry in entries.items() if entry is not None
    }


def _held_facts(
    reader: NodeViewReader, dcid: str, *, catalogue: Catalogue, deadline: float
) -> _Facts | None:
    """`dcid`'s facts where `reader` holds the node by the node view's rule, else `None`. A node
    the `name` and `typeOf` read answers with neither is read again for every outgoing arc
    (`->*`), capped at the node view's `CENTRE_MAX_BYTES`, and held when an arc other than an
    unresolved target is left (`schema_document.resolved_entry`), as the node view finds its
    centre. Such a node has no label, types or `declared_by`.

    Raises:
        GraphCapExceededError: the `->*` read passed its cap, where the node view refuses too.
    """
    found = _read_facts(reader, [dcid], catalogue=catalogue, deadline=deadline).get(dcid)
    if found is not None:
        return found
    arcs = reader.node_all_arc_targets_many([dcid], max_bytes=CENTRE_MAX_BYTES, deadline=deadline)
    if resolved_entry(arcs.get(dcid)) is None:
        return None
    return _Facts(label=None, types=[], declared_by=None)


def _read_dcid(
    readers: _Readers, dcid: str, *, catalogue: Catalogue, deadline: float
) -> _Facts | None:
    """`dcid`'s facts as the node view route finds the node, `None` where no read holds it."""
    if is_un_dcid(dcid):
        if readers.un is None:
            return None
        return _held_facts(readers.un, dcid, catalogue=catalogue, deadline=deadline)
    held = _held_facts(readers.mixer, dcid, catalogue=catalogue, deadline=deadline)
    if held is not None or readers.un is None:
        return held
    return _held_facts(readers.un, dcid, catalogue=catalogue, deadline=deadline)


def _kept_candidates(body: dict) -> list[IndicatorCandidate]:
    """The first `INDICATOR_CANDIDATES` valid candidates of an indicator resolve answer that are
    typed `StatisticalVariable` or name no type."""
    candidates = parse_indicator_candidates(body, skip_invalid=True)
    kept = [c for c in candidates if not c.types or _STATVAR in c.types]
    return kept[:INDICATOR_CANDIDATES]


def _resolve_candidates(
    indicators: IndicatorResolver, query: str, *, target: str, un: bool, deadline: float
) -> list[IndicatorCandidate]:
    body = indicators.resolve_indicators(query, target=target, un=un, deadline=deadline)
    return _kept_candidates(body)


async def _within[R](awaitable: Awaitable[R], deadline: float) -> R:
    async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
        return await awaitable


def _log_failure(what: str, query: str, exc: Exception) -> None:
    """One warning naming the failed part and the query, with a traceback for a failure outside
    `_EXPECTED_FAILURES` and `_LegUnavailable`."""
    expected = isinstance(exc, (*_EXPECTED_FAILURES, _LegUnavailable))
    logger.warning(
        "schema search %s failed for %r: %r", what, query, exc, exc_info=None if expected else exc
    )


def _rate_limited(exc: Exception) -> bool:
    return isinstance(exc, GraphReadError) and exc.upstream_status == _RATE_LIMITED


@dataclass(slots=True)
class _Leg:
    """One leg: the source it answers for and the coroutine function that runs it."""

    source: SearchSource
    run: Callable[[], Awaitable[list[SchemaSearchHit]]]


@dataclass(frozen=True, slots=True)
class _HitFacts:
    """The label reads' answer: the facts found, by ref, the refs of every batch that answered,
    and whether every batch answered."""

    found: dict[str, _Facts]
    checked: frozenset[str]
    answered: bool


async def _read_hit_facts(
    search: SchemaSearchDeps,
    readers: _Readers,
    refs: Sequence[str],
    *,
    query: str,
    catalogue: Catalogue,
    deadline: float,
) -> _HitFacts:
    """The facts of `refs` from one read per deployment: UN ids from the UN deployment, where
    the request reads it, and the rest from the mixer. A read that fails leaves its refs
    unchecked and the other read's facts stand."""
    batches: list[tuple[NodeViewReader, list[str]]] = [
        (readers.mixer, [ref for ref in refs if not is_un_dcid(ref)])
    ]
    if readers.un is not None:
        batches.append((readers.un, [ref for ref in refs if is_un_dcid(ref)]))
    batches = [(reader, part) for reader, part in batches if part]
    outcomes = await asyncio.gather(
        *(
            _within(
                run_dcp_read(
                    search.executor,
                    _read_facts,
                    reader,
                    part,
                    catalogue=catalogue,
                    deadline=deadline,
                ),
                deadline,
            )
            for reader, part in batches
        ),
        return_exceptions=True,
    )
    found: dict[str, _Facts] = {}
    checked: set[str] = set()
    answered = True
    for (_, part), outcome in zip(batches, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):
                raise outcome
            _log_failure("label read", query, outcome)
            answered = False
            continue
        found |= outcome
        checked.update(part)
    return _HitFacts(found=found, checked=frozenset(checked), answered=answered)


def _with_facts(
    hits: Mapping[str, SchemaSearchHit], facts: _HitFacts
) -> dict[str, SchemaSearchHit]:
    """`hits` with the facts found, less each indicator hit a read that answered did not find,
    or found with no `StatisticalVariable` type where the resolver named none."""
    kept: dict[str, SchemaSearchHit] = {}
    for ref, hit in hits.items():
        found = facts.found.get(ref)
        dead = found is None or (not hit.types and _STATVAR not in found.types)
        if hit.via == "indicator" and ref in facts.checked and dead:
            continue
        if found is not None and hit.via != "dcid":
            hit = hit.model_copy(
                update={
                    "label": found.label or hit.label,
                    "types": found.types or hit.types,
                    "declared_by": found.declared_by,
                }
            )
        kept[ref] = hit
    return kept


async def _search(
    deps: GraphDeps,
    search: SchemaSearchDeps,
    request: Request,
    query: str,
    *,
    base: bool,
    un: bool,
    started: float,
) -> SchemaSearch:
    """The answer to `query`, its deadlines counted from `started`.

    Raises:
        SchemaSearchRateLimited: every leg failed, each on a graph read answered HTTP 429.
        SchemaSearchUnavailable: every leg failed.
    """
    leg_deadline = started + SCHEMA_SEARCH_LEG_SECONDS
    deadline = started + SCHEMA_SEARCH_TIMEOUT_SECONDS
    catalogue = current_catalogue()
    reads_un = un and catalogue.un_system_enabled
    views = deps.node_views
    readers = (
        None
        if views is None
        else _Readers(mixer=views.mixer, un=views.un_reader if reads_un else None)
    )

    async def dcid_leg() -> list[SchemaSearchHit]:
        if readers is None:
            raise _LegUnavailable("the surface has no node view reader")
        facts = await run_dcp_read(
            search.executor,
            _read_dcid,
            readers,
            query,
            catalogue=catalogue,
            deadline=leg_deadline,
        )
        if facts is None:
            return []
        return [
            SchemaSearchHit(
                ref=query,
                label=facts.label,
                types=facts.types,
                declared_by=facts.declared_by,
                via="dcid",
            )
        ]

    async def names_leg() -> list[SchemaSearchHit]:
        async with search.names:
            response = await run_resolve(
                deps,
                ResolveRequest(texts=[query]),
                policy=published_membership(request),
                deadline=leg_deadline,
                near_matches=False,
            )
        (result,) = response.results
        if result.status == "resolved" and result.entity is not None:
            entities = [result.entity]
        elif result.status == "ambiguous":
            entities = result.candidates or []
        else:
            entities = []
        return [
            SchemaSearchHit(ref=e.ref, label=e.label, types=[], declared_by=None, via="name")
            for e in entities
        ]

    def indicator_leg(
        *, target: str, un_deployment: bool
    ) -> Callable[[], Awaitable[list[SchemaSearchHit]]]:
        async def run() -> list[SchemaSearchHit]:
            if search.indicators is None:
                raise _LegUnavailable("the surface has no indicator resolver")
            candidates = await run_dcp_read(
                search.executor,
                _resolve_candidates,
                search.indicators,
                query,
                target=target,
                un=un_deployment,
                deadline=leg_deadline,
            )
            return [
                SchemaSearchHit(
                    ref=c.dcid,
                    label=None,
                    types=[_STATVAR] if c.types else [],
                    declared_by=None,
                    via="indicator",
                )
                for c in candidates
            ]

        return run

    legs: list[_Leg] = []
    if _DCID.fullmatch(query):
        legs.append(_Leg(source="dcid", run=dcid_leg))
    if len(query) <= RESOLVE_TEXT_MAX_LENGTH:
        legs.append(_Leg(source="names", run=names_leg))
    legs.append(_Leg(source="dfo", run=indicator_leg(target="custom_only", un_deployment=False)))
    if base:
        legs.append(_Leg(source="base", run=indicator_leg(target="base_only", un_deployment=False)))
    if reads_un:
        legs.append(
            _Leg(
                source="un_system",
                run=indicator_leg(target="custom_only", un_deployment=True),
            )
        )

    outcomes = await asyncio.gather(
        *(_within(leg.run(), leg_deadline) for leg in legs), return_exceptions=True
    )
    hits: dict[str, SchemaSearchHit] = {}
    unavailable: list[SearchSource] = []
    failures: list[Exception] = []
    for leg, outcome in zip(legs, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):
                raise outcome
            _log_failure(f"leg {leg.source}", query, outcome)
            unavailable.append(leg.source)
            failures.append(outcome)
            continue
        for hit in outcome:
            hits.setdefault(hit.ref, hit)
    if len(failures) == len(legs):
        if all(_rate_limited(exc) for exc in failures):
            waits = [
                exc.retry_after
                for exc in failures
                if isinstance(exc, GraphReadError) and exc.retry_after is not None
            ]
            raise SchemaSearchRateLimited(max(waits, default=None))
        raise SchemaSearchUnavailable(unavailable)

    unread = [ref for ref, hit in hits.items() if hit.via != "dcid"]
    if unread:
        if readers is None:
            unavailable.append("labels")
        else:
            facts = await _read_hit_facts(
                search, readers, unread, query=query, catalogue=catalogue, deadline=deadline
            )
            hits = _with_facts(hits, facts)
            if not facts.answered:
                unavailable.append("labels")
    return SchemaSearch(query=query, hits=list(hits.values()), unavailable=unavailable)


async def get_schema_search(
    deps: GraphDepsDep,
    request: Request,
    q: Annotated[SearchQuery, Query()],
    base: bool = False,
    un: bool = False,
) -> JSONResponse:
    started = time.monotonic()
    search = deps.schema_search
    if search is None:
        raise GraphDepsNotSet()
    await deps.generation.refresh_if_stale()
    catalogue = current_catalogue()
    reads_un = un and catalogue.un_system_enabled
    key = SearchKey(
        stamp=deps.generation.stamp().key,
        un_stamp=search.un_generation.stamp().key if reads_un else None,
        release=served_origin_key(),
        origin_catalogue=catalogue.build_id,
        un_system_enabled=catalogue.un_system_enabled,
        query=q,
        base=base,
        un=un,
    )
    answer = search.cache.get(key)
    if answer is None:
        answer = await _search(deps, search, request, q, base=base, un=un, started=started)
        if not answer.unavailable:
            search.cache.put(key, answer)
    return JSONResponse(answer.model_dump(mode="json"))


def classify_schema_search_exception(exc: BaseException) -> ProblemSpec | None:
    """Both refusals as retryable 503 `service_unavailable`, a rate-limited one with
    `cause: {"code": "upstream_rate_limited"}` and `Retry-After`, as the schema routes answer a
    rate-limited read."""
    if isinstance(exc, SchemaSearchRateLimited):
        retry_after = (
            exc.retry_after
            if exc.retry_after is not None
            else GRAPH_RATE_LIMITED_RETRY_AFTER_SECONDS
        )
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=str(exc),
            retryable=True,
            cause=Cause(code="upstream_rate_limited"),
            headers={"Retry-After": str(math.ceil(retry_after))},
            log="Graph schema search's every leg was answered 429",
        )
    if isinstance(exc, SchemaSearchUnavailable):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=str(exc),
            retryable=True,
            log=f"Graph schema search answered with no leg: {exc}",
        )
    return None


def build_schema_search_door() -> DoorContribution:
    router = APIRouter()
    # `response_model` documents the body in OpenAPI. The JSONResponse passes through.
    router.add_api_route(
        "/api/graph/schema/search",
        get_schema_search,
        methods=["GET"],
        response_model=SchemaSearch,
    )
    return DoorContribution(
        router=router,
        classify=classify_schema_search_exception,
        exception_types=(SchemaSearchUnavailable, SchemaSearchRateLimited),
        exception_exemplars=(
            SchemaSearchUnavailable(["names", "dfo"]),
            SchemaSearchRateLimited(None),
        ),
    )
