"""Serve the deployment's schema document, the list of the DFO graph's Classes and
Properties, and the list of those the UN System Data Commons declares.

`GET /api/graph/schema` serves `SchemaDocument` and `GET /api/graph/schema/terms`
serves `SchemaTerms`, every Class and Property the mixer lists by `typeOf` and none of the UN
deployment's. One build (`graph_api.schema_document.build_schema`) yields both,
and `GraphDeps.schema` holds its result in a single-flight cache of two entries, keyed by
the graph generation token, the content hash of the code index the build read and the build of
the origin catalogue whose provenance table names origins for `declared_by`, beside the cache's
own generation stamp. Two entries let a request stamped with an unknown generation
keep the entry of the known one. The lifespan registers it as `graph_api.schema`, so a
generation change clears it.

A request refreshes a stale generation stamp, waiting up to
`SCHEMA_GENERATION_WAIT_SECONDS` (3 s) for the read so an instance back from an idle spell
keys on the known generation and keeps its built entry, then waits for the entry until
`GRAPH_NODE_TIMEOUT_SECONDS` from its start. A miss runs the whole build as one job on the
DCP-read executor, holding one worker, under a flight deadline of
`SCHEMA_FLIGHT_TIMEOUT_SECONDS` from submission. Concurrent requests share the flight, and a
request whose own deadline passes first answers retryable 503 `service_unavailable` with
`Retry-After: 5` and `cause: {"code": "schema_building"}` while the flight runs on and stores
its result for the next request. A flight
that outlives its deadline stores nothing, and its waiters answer 504 `timeout`. An executor
with every worker busy refuses the build with `DcpExecutorSaturated`, which the surface answers
as retryable 503. A read past its deadline answers 504 `timeout`. A listing that reached its
cap, or a document with no node DFO declares, is a server fault and answers non-retryable 500
`internal_error`. A DCP 429 answers retryable 503 with `Retry-After: 5`, any other DCP 4xx 502
`upstream_error`, and any other read failure retryable 503. Under a known generation the entry
keeps either refusal as `RefusedSchema` for
`SCHEMA_REFUSAL_TTL_SECONDS` (60 s), so every request for the key in that time answers the
same 500 with no build, and the first request after it builds again. Under an unknown
generation the build raises the refusal unkept and the next request builds again. Any other
failed build is not kept.

The cached value holds each body encoded once in the models' field order, so two builds of one
graph give the same bytes. Each answers `public, max-age=300` with a strong `ETag` over those
bytes and a gzip representation, and answers `If-None-Match` with 304
(`dfo_platform.static_response.FrozenJSON`).

`POST /api/graph/schema/check` takes 1 to `CHECK_MAX_ROWS` candidate StatVar definitions,
checks them in order from the first, and answers the faults `graph_api.schema_check.check_rows`
finds in the rows it checked, how many rows that is (`rows_checked`), and the generation of the
build it checked against. The entry keeps, beside its frozen bodies, the build's
`SchemaLookup`, built once per entry. The check compares served definitions against the code
index the request computed the entry's key from, which is the index the build read. The
check's deadline is `GRAPH_SCHEMA_CHECK_TIMEOUT_SECONDS` from its start. It waits on the same
flight as the GET routes until that deadline and answers their 503 and 504 the same way, so the
first check after a graph change builds the document before it plans its reads.

A body outside the request model's envelope and size bounds is refused 400 `request_invalid`:
`rows` missing, empty or longer than `CHECK_MAX_ROWS`, an unknown member, a row past
`CHECK_MAX_CONSTRAINT_PROPERTIES` or `CHECK_MAX_OBSERVATION_PROPERTIES`, a string past 512
characters, or a constraint value that is a number past plus or minus 2**53, `NaN` or an
infinity. The pointer names the first such member, `/rows/37/constraint_properties` when it is
in row 37. Everything else about a row's content is a finding. With the document in hand,
`plan_reads` decides on the event loop, before anything is submitted, how many rows the check
takes and which ids they read: rows from the first while their ids stay within
`CHECK_MAX_READ_IDS`, the first row always. Whether the bound catalogue serves the UN origin
decides whether a UN id is read or gets `origin_not_served`.

Every check runs as one job on `GraphDeps.check_executor`, whose
`GRAPH_SCHEMA_CHECK_CONCURRENCY` (2) workers are its own, so a check takes none of the
shared DCP-read workers. The job reads, checks and encodes the response, and the event loop
only wraps its bytes. A worker stays held until its job returns, even after its request is
cancelled or times out, so at most two checks run at once in the process. One more is refused at
submit and answers retryable 503 `service_unavailable` with `Retry-After: 1`
(`CHECK_BUSY_RETRY_AFTER_SECONDS`) and `cause: {"code": "check_busy"}` (`SchemaCheckBusy`). A read past the check's deadline or a failed read answers as the GET
routes answer it, with no findings. The request stops waiting at the check's deadline and
answers 504 `timeout` while its job runs on, so a read that answers late yields no 200. The job
stops before the row whose findings would pass `CHECK_MAX_FINDINGS`, or take the encoded
response past `CHECK_MAX_BODY_BYTES` (4 MB), and always takes the first row. A caller resumes at
`rows_checked` when it is less than the rows it sent.

`GET /api/graph/schema/usage` serves `DatasetUsage`, built by
`graph_api.dataset_usage.build_usage` from the bound code index, the served coverage index and
one batched read of every Provenance node's `name` and `description`, each empty answer
confirmed alone, so each Provenance's label and description are the graph's.
`GraphDeps.dataset_usage`, a single-flight cache of one entry keyed by the code index's content
hash and the coverage build (`CoverageManifest.build_id`), holds the frozen body, and the
lifespan registers it as `graph_api.dataset_usage`, so a generation change clears it too. A
miss reads and builds the body on the cache's own one-worker executor under the flight
deadline, and concurrent misses share that build. A request whose wait passes
`GRAPH_NODE_TIMEOUT_SECONDS` answers retryable 503 with `Retry-After: 5` and
`cause: {"code": "schema_building"}`, a build past its flight deadline answers 504, and a
failed read answers as it does for the other two routes. A failed build is not kept.

`GET /api/graph/schema/terms/un_system` serves `SchemaTerms` for the Classes and Properties
the UN System Data Commons declares, built by `graph_api.un_terms.build_un_terms` from two
listings read through the UN deployment's own client (`GraphDeps.un_terms`), past the origin
client's routing, which would send `Class` and `Property` to the mixer. That client skips the
catalogue's `un_system_enabled` gate too, so the route checks it first. With the UN origin off,
or on a surface with no UN reader, it answers 404 `not_found`, not retryable, with
`cause: {"code": "origin_not_served"}`, and reads nothing. The refusal goes out `no-store`, so no
cache keeps it past a change of the flag. A 200 body goes out `public, max-age=300`, so a
browser or CDN can serve it for up to 5 minutes after the flag is switched off. The body sits in
a single-flight cache of one entry on an executor of one worker, under the one key
`UN_TERMS_KEY` and the cache's generation stamp, the UN deployment's
(`origin_catalogue.graph.UnGeneration`, the bound catalogue's build and the UTC date). The UN
deployment serves no generation token, so a UN change shows on the next UTC day or catalogue
build, and the body's `generation` is `null`. `warm_un_terms` builds the body in the Graph
lifespan. A miss builds under the flight deadline. Before any body is served, a request waits
until `GRAPH_NODE_TIMEOUT_SECONDS` from its start, and the waits and failures answer as on the
usage route. Once one is served (`UnTerms.served`), a miss starts or joins the new build and
answers the served body at once, as does a failed build or a kept refusal, so the UTC rollover
and a catalogue build answer no 503. The build waits out 429s and repeated page tokens on the
page they hit, within the flight (`graph_api.un_terms`). The entry keeps three refusals:

- a listing past its cap, answered as on `GET /api/graph/schema`, for
  `SCHEMA_REFUSAL_TTL_SECONDS`;
- a listing with no UN term (`UnTermsListingEmptyError`), answered 503, retryable, with
  `cause: {"code": "listing_incomplete"}`, for `SCHEMA_REFUSAL_TTL_SECONDS`;
- a listing still rate-limited (`UnTermsRateLimitedError`), answered 503, retryable, with
  `cause: {"code": "upstream_rate_limited"}`, for `SCHEMA_REFUSAL_TTL_SECONDS` or the UN front
  door's `Retry-After` where that is longer.

Each 503 sends `Retry-After` with the seconds left on the refusal. Any other failed build is not
kept.

`GET /api/graph/schema/usage/base` and `GET /api/graph/schema/usage/un_system` serve the
`DatasetUsage` body of one outside origin, with its `terms`, built by
`graph_api.origin_usage.build_origin_usage` from the bound origin catalogue and graph release,
whatever the request's source scope, with no graph read. `GraphDeps.origin_usage`, a
single-flight LRU cache of two entries keyed by the origin and the release id
(`graph_api.origin_usage.OriginUsageKey`) under the catalogue build's stamp
(`graph_api.origin_usage.CatalogueGeneration`), holds each frozen body, built on its own
one-worker executor. A request bound to another build or release builds its own entry, which
evicts the least recently used one. A request's wait and a build past its flight deadline
answer as on the usage route, and each build logs one line with its origin, catalogue build,
release, time and encoded size. With the UN origin off, `GET /api/graph/schema/usage/un_system`
answers 404 `not_found`, not retryable, with `cause: {"code": "origin_not_served"}`, and builds
nothing. A bound catalogue that is not a loaded `CatalogueSnapshot` holds no tables to build
from, and both routes answer it the same way.

`GET /api/graph/schema/node/{dcid}` serves `SchemaNodeView`, one node and its schema
neighbours, built by `graph_api.node_view.build_node_view` from at most eight reads, and for a
Class one more per level below its narrower classes. Its flights run on an executor of
`NODE_VIEW_FLIGHT_WORKERS` workers that admits one flight per worker, so a miss that finds every
worker busy answers retryable 503 at once. Their reads run on a pool of
`NODE_VIEW_FLIGHT_WORKERS * READS_AT_ONCE` workers, apart from the DCP-read executor. A dcid past
`DCID_MAX_LENGTH` answers 400 `request_invalid` before any read. A UN id
(`origin_catalogue.graph.is_un_dcid`) is read from the UN deployment's own client once the route
has checked that the bound catalogue serves the UN origin; with the origin off, or on a surface
with no UN reader, it answers 404 `not_found`, not retryable, with
`cause: {"code": "origin_not_served"}`, and reads nothing. Any other id is read from the mixer's
own client, and a node the mixer does not hold, such as `unCode`, is read from the UN deployment
when the UN origin is served. A node neither read holds answers 404 `not_found`, not retryable,
with `cause: {"code": "node_not_held"}`. The centre's `->*`, a closure or a chain edge read
past the view's byte cap answers 500 `internal_error`, not retryable, with
`cause: {"code": "node_view_too_large"}`, and a paged read that repeats its continuation token
or continues after its page cap the same with `cause: {"code": "graph_not_advancing"}`. The
UN deployment is not read for either. All refusals go out `no-store`. The route binds no code
registry endpoint, so it refuses with platform codes and names its reason in `cause.code`, where
`GET /api/graph/node/{dcid}` answers the Graph code `node_not_held` at the top level.

`NodeViews` holds two caches per deployment. The view caches hold at most
`NODE_VIEW_CACHE_SIZE` entries and `NODE_VIEW_CACHE_BYTES` of encoded bodies each: the mixer's
keyed by `NodeViewKey` under the mixer's generation, so a generation change clears it, and the
UN deployment's under `origin_catalogue.graph.UnGeneration`, whose body `generation` is `null`.
Beside each, a miss cache of `NODE_VIEW_MISS_CACHE_SIZE` dcids keeps the deployment's
refusals (`NodeViewMiss`), so a made-up dcid evicts no view and is answered again with no
read: the mixer's for its generation, the UN deployment's for `UN_NODE_VIEW_MISS_TTL_SECONDS`.
Each flight's deadline is `GRAPH_NODE_TIMEOUT_SECONDS` from its start, so a UN read after a
mixer miss gets a deadline of its own, and a request waits `GRAPH_NODE_TIMEOUT_SECONDS` from its
start over both. Either passing answers 504 `timeout`. A read the graph answers with HTTP 429
after the view's own waits answers retryable 503 `service_unavailable` with
`Retry-After: 5`, on this route as on the others. Any other failed read answers as on the other
routes. Each view answers `public, max-age=300` with a strong `ETag`, gzip and 304 as the
document does.

After each build the door logs one line timing the encoding of the document and terms bodies
and giving their sizes, beside the build's own per-phase line.

`warm_schema` builds the entry for the current generation and the bound code index before an
instance takes traffic. The Graph lifespan awaits it for its live reader. An unfinished build
logs and the first request joins it. A failed build logs and the first request builds again,
except a refusal the entry keeps, which a request within its lifetime answers 500 with no
build.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import NamedTuple

from fastapi import APIRouter, Request, Response

from code_index import CodeIndex
from code_index.shared import get_index
from dc_graph import GraphCapExceededError, GraphNotAdvancingError
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.problem import Cause
from dfo_platform.refusal import SERVER_ERROR_FALLBACK_CODE, TIMEOUT_DEFAULT
from dfo_platform.single_flight_cache import SingleFlightCache, WaitTimeout
from dfo_platform.static_response import FrozenJSON, freeze_public_json
from graph_api.config import (
    GRAPH_NODE_TIMEOUT_SECONDS,
    GRAPH_SCHEMA_CHECK_CONCURRENCY,
    GRAPH_SCHEMA_CHECK_TIMEOUT_SECONDS,
    SCHEMA_FLIGHT_TIMEOUT_SECONDS,
)
from graph_api.dataset_usage import PROVENANCE_NODE_PROPERTIES, ProvenanceCoverage, build_usage
from graph_api.deps import (
    DcpExecutorSaturated,
    GraphDeps,
    GraphDepsDep,
    GraphDepsNotSet,
    NodeViews,
    log_queue_wait,
    run_dcp_read,
)
from graph_api.doors.node import DcidTooLong, graph_read_problem
from graph_api.node_view import NodeNotHeldError, NodeViewReader, build_node_view
from graph_api.node_view_models import SchemaNodeView
from graph_api.origin_usage import OriginUsageKey, UsageOrigin, build_origin_usage
from graph_api.reader import GraphNodeReader
from graph_api.routing import DCID_MAX_LENGTH
from graph_api.schema_check import SchemaLookup, check_rows, plan_reads
from graph_api.schema_document import (
    SchemaListingTruncatedError,
    SchemaNoDfoDeclarationError,
    build_schema,
)
from graph_api.schema_models import (
    DatasetUsage,
    SchemaCheckRequest,
    SchemaCheckResponse,
    SchemaCheckRow,
    SchemaDocument,
    SchemaTerms,
)
from graph_api.un_terms import (
    UnTermsListingEmptyError,
    UnTermsRateLimitedError,
    build_un_terms,
)
from graph_generation import GenerationProvider, GenerationStamp
from graph_generation.stamp import UNKNOWN_KEY
from graph_release.context import served_origin_key
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from origin_catalogue import Catalogue, CatalogueSnapshot, current_catalogue
from origin_catalogue.graph import is_un_dcid

logger = logging.getLogger(__name__)

SCHEMA_RETRY_AFTER_SECONDS = 5
"""The `Retry-After` a request sends when its deadline passes while the build runs on."""

CHECK_BUSY_RETRY_AFTER_SECONDS = 1
"""The `Retry-After` a check refused for a full check executor sends. A check of
`CHECK_MAX_READ_IDS` absent ids took 2.6 to 4.4 s on the dev graph and one of 41 under 1 s, so
a worker usually frees within a second."""

SCHEMA_GENERATION_WAIT_SECONDS = 3.0
"""How long a schema request waits for a stale generation read. Past it the stamp is unknown
and the request builds a second entry under `UNKNOWN_KEY`, which costs a whole build; a cold
read of the generation took 1.5 s from a laptop and 1.9 s at an instance's start."""

GRAPH_RATE_LIMITED_RETRY_AFTER_SECONDS = 5
"""The `Retry-After` a request sends when a graph read answered HTTP 429 after the reads' own
waits."""

SCHEMA_REFUSAL_TTL_SECONDS = 60.0
"""How long the schema cache keeps a build's refusal (`RefusedSchema`). A listing the upstream
answers empty or truncated for a moment refuses the requests of this window, and the first
request after it builds again, so a lasting fault costs one build a minute."""

_MAX_AGE_SECONDS = 300
_RATE_LIMITED = 429


class SchemaKey(NamedTuple):
    """The graph generation token, `UNKNOWN_KEY` where it is unknown, the content hash of the
    code index whose StatVar rows the build read, and the build of the origin catalogue whose
    provenance table names the origins of `declared_by`."""

    generation: str
    code_index: str
    origin_catalogue: str


@dataclass(frozen=True, slots=True)
class FrozenSchema:
    """Both bodies of one build, each encoded once, the build's generation and the lookup the
    check reads."""

    document: FrozenJSON
    terms: FrozenJSON
    generation: str | None
    lookup: SchemaLookup


@dataclass(frozen=True, slots=True)
class RefusedSchema:
    """A build refusal: a truncated term listing, a document with no node DFO declares, or a UN
    terms build whose listing was empty or still rate-limited. The cache keeps it for `ttl_s`,
    `SCHEMA_REFUSAL_TTL_SECONDS` unless the UN front door's `Retry-After` names longer. `error`
    makes a fresh exception for each request it answers."""

    error: Callable[[], GraphReadError]
    ttl_s: float = SCHEMA_REFUSAL_TTL_SECONDS


type SchemaEntry = FrozenSchema | RefusedSchema


class SchemaBuildPending(Exception):
    """A request's deadline passed while the build it waits on runs on."""

    def __init__(self) -> None:
        super().__init__("the schema is still being built; retry")


class SchemaBuildExpired(Exception):
    """The build outlived its flight deadline, so it stored nothing."""

    def __init__(self) -> None:
        super().__init__("the schema build passed its deadline")


class SchemaCheckBusy(Exception):
    """Every worker of `GraphDeps.check_executor` holds a check job."""

    def __init__(self) -> None:
        super().__init__(
            f"{GRAPH_SCHEMA_CHECK_CONCURRENCY} schema checks are already running; retry"
        )


class SchemaReadFailed(Exception):
    """A graph read of the build or of the check failed with `error`."""

    def __init__(self, error: GraphReadError) -> None:
        super().__init__(f"a schema graph read failed: {error}")
        self.error = error


class UnOriginNotServed(Exception):
    """The bound catalogue does not serve the UN origin, or the surface has no UN reader."""

    def __init__(self) -> None:
        super().__init__("this deployment does not serve the UN System Data Commons origin")


class OriginUsageNotServed(Exception):
    """The bound catalogue is not a loaded `CatalogueSnapshot`, so it holds no tables to build
    an origin's usage body from."""

    def __init__(self, origin: UsageOrigin) -> None:
        super().__init__(f"this deployment holds no origin catalogue to serve {origin} usage from")


UN_TERMS_KEY = "un_system"
"""The one key of the UN terms cache, whose own generation stamp is the UN deployment's."""

type UnTermsEntry = FrozenJSON | RefusedSchema


class NodeViewKey(NamedTuple):
    """The dcid of a node view, the build of the origin catalogue that names its origins and
    whether that catalogue serves the UN origin. Each cache adds its own generation stamp: the
    mixer's for a view read from the mixer, the UN deployment's for one read from there."""

    dcid: str
    origin_catalogue: str
    un_system_enabled: bool


NODE_VIEW_CACHE_SIZE = 2_000
"""The most node views each node view cache keeps, the mixer's and the UN deployment's."""

NODE_VIEW_CACHE_BYTES = 64 * 1024 * 1024
"""The most bytes of encoded bodies each node view cache keeps, identity and gzip together, the
least recently used leaving first, and a body past it is served and not kept. A view with 6,002
neighbours weighs 2.49 MB (2.44 MB identity). A view has no fixed upper weight, since the
centre's `other_properties` keep up to 100 node targets for each of any number of properties,
so the bound is this budget. The two caches together hold at most 128 MiB of the 1.6 GiB the
dev apis service leaves beside two origin catalogue builds (deploy/origin-catalogue.md)."""

NODE_VIEW_MISS_CACHE_SIZE = 1_000
"""The most refused dcids each node view miss cache keeps, apart from the views, so made-up dcids
evict no view."""

UN_NODE_VIEW_MISS_TTL_SECONDS = 300.0
"""How long the UN deployment's miss cache keeps a refusal. The UN deployment serves no
generation token, so a node it gains shows within this time."""

NODE_VIEW_FLIGHT_WORKERS = 4
"""The workers the node view flights run on, one flight each. A miss that finds them all busy is
refused."""


class NodeViewMiss(Enum):
    """A view's refusal, kept in a miss cache apart from the views. `NOT_HELD`: the deployment
    answered the dcid with no arc. `TOO_LARGE`: the centre's `->*` answered past
    `NODE_VIEW_MAX_BYTES`, or a closure or chain edge read did. `NOT_ADVANCING`: a paged read
    repeated its continuation token or continued after its page cap
    (`dc_graph.GraphNotAdvancingError`), which a retry pages the same way."""

    NOT_HELD = "not_held"
    TOO_LARGE = "too_large"
    NOT_ADVANCING = "not_advancing"


type NodeViewEntry = FrozenJSON | NodeViewMiss


class NodeViewTooLarge(Exception):
    """A read of the view of `dcid` answered past the view's byte cap, so the node has no view."""

    def __init__(self, dcid: str) -> None:
        super().__init__(f"a graph read for the view of {dcid!r} passed its byte cap")
        self.dcid = dcid


class NodeViewNotAdvancing(Exception):
    """A paged read of the view of `dcid` repeated its continuation token or continued after
    its page cap, so the node has no view."""

    def __init__(self, dcid: str) -> None:
        super().__init__(f"a paged graph read for the view of {dcid!r} did not advance")
        self.dcid = dcid


class NodeViewTimedOut(Exception):
    """A node view's wait or its flight passed its deadline."""

    def __init__(self) -> None:
        super().__init__("the node view's graph reads passed their deadline; retry")


class UsageKey(NamedTuple):
    """The content hash of the code index and the coverage build the usage body reads."""

    code_index: str
    coverage_build: str


def new_schema_cache[K, V](
    *,
    entries: int,
    executor: Executor,
    generation: GenerationProvider,
    clock: Callable[[], float] = time.monotonic,
) -> SingleFlightCache[K, V]:
    """A cache of this door's bodies holding `entries` entries, each built on `executor` within
    `SCHEMA_FLIGHT_TIMEOUT_SECONDS`, with a key that carries `generation`'s stamp and entry
    lifetimes measured on `clock`. `GraphDeps.schema` holds two, so a request stamped with an
    unknown generation keeps the entry of the known one, `GraphDeps.dataset_usage` holds
    one, the UN terms cache one, and `GraphDeps.origin_usage` two, one per origin."""
    return SingleFlightCache(
        maxsize=entries,
        flight_timeout_s=SCHEMA_FLIGHT_TIMEOUT_SECONDS,
        executor=executor,
        generation=generation,
        clock=clock,
    )


def _view_bytes(entry: NodeViewEntry) -> int:
    """The bytes `entry`'s encoded bodies hold, identity and gzip."""
    if isinstance(entry, NodeViewMiss):
        return 0
    return len(entry.identity) + len(entry.gzip or b"")


def new_node_view_cache(
    *, executor: Executor, generation: GenerationProvider
) -> SingleFlightCache[NodeViewKey, NodeViewEntry]:
    """A cache of at most `NODE_VIEW_CACHE_SIZE` node views and `NODE_VIEW_CACHE_BYTES` of their
    bodies, each built on `executor` within `GRAPH_NODE_TIMEOUT_SECONDS`, with a key that
    carries `generation`'s stamp."""
    return SingleFlightCache(
        maxsize=NODE_VIEW_CACHE_SIZE,
        flight_timeout_s=GRAPH_NODE_TIMEOUT_SECONDS,
        executor=executor,
        generation=generation,
        weigh=_view_bytes,
        max_weight=NODE_VIEW_CACHE_BYTES,
    )


def new_node_view_miss_cache(
    *, generation: GenerationProvider, ttl_s: float | None = None
) -> SingleFlightCache[NodeViewKey, NodeViewMiss]:
    """A cache of `NODE_VIEW_MISS_CACHE_SIZE` view refusals with a key that carries
    `generation`'s stamp, each kept for the stamp's own lifetime, or for `ttl_s` when given.
    Refusals join it by `offer`, so it needs no executor."""

    def ttl(stamp: GenerationStamp) -> float:
        return stamp.max_age_s if ttl_s is None else ttl_s

    return SingleFlightCache(
        maxsize=NODE_VIEW_MISS_CACHE_SIZE,
        flight_timeout_s=GRAPH_NODE_TIMEOUT_SECONDS,
        generation=generation,
        ttl_for_stamp=ttl,
    )


def _entry_ttl(entry: SchemaEntry | UnTermsEntry) -> float | None:
    """A refusal's own `ttl_s`, and the cache's own TTL for built bodies."""
    return entry.ttl_s if isinstance(entry, RefusedSchema) else None


def _freeze(model: SchemaDocument | SchemaTerms | DatasetUsage | SchemaNodeView) -> FrozenJSON:
    return freeze_public_json(model.model_dump(mode="json"), max_age=_MAX_AGE_SECONDS)


def _request_deadline(seconds: float | None = None) -> float:
    """`seconds` from now, `GRAPH_NODE_TIMEOUT_SECONDS` when `None`."""
    return time.monotonic() + (GRAPH_NODE_TIMEOUT_SECONDS if seconds is None else seconds)


def _build_entry(
    deps: GraphDeps,
    *,
    code_index: CodeIndex,
    catalogue: Catalogue,
    generation: str | None,
    deadline: float,
) -> SchemaEntry:
    """Build and freeze the bodies, or return a refusal for the cache to keep for
    `SCHEMA_REFUSAL_TTL_SECONDS`. Under an unknown generation, `generation` `None`, the refusal
    is raised and not kept."""
    try:
        schema = build_schema(
            reader=deps.graph,
            code_index=code_index,
            catalogue=catalogue,
            generation=generation,
            deadline=deadline,
        )
    except SchemaListingTruncatedError as exc:
        if generation is None:
            raise
        return RefusedSchema(error=partial(SchemaListingTruncatedError, str(exc)))
    except SchemaNoDfoDeclarationError as exc:
        if generation is None:
            raise
        return RefusedSchema(error=partial(SchemaNoDfoDeclarationError, exc.node_count))
    started = time.monotonic()
    document = _freeze(schema.document)
    document_at = time.monotonic()
    terms = _freeze(schema.terms)
    ended = time.monotonic()
    logger.info(
        "schema encode took %d ms: document %d ms %d bytes; terms %d ms %d bytes",
        round((ended - started) * 1_000),
        round((document_at - started) * 1_000),
        len(document.identity),
        round((ended - document_at) * 1_000),
        len(terms.identity),
    )
    return FrozenSchema(
        document=document,
        terms=terms,
        generation=schema.document.generation,
        lookup=SchemaLookup.of(
            schema.document,
            unresolved_types=schema.unresolved_types,
            unresolved_parents=schema.unresolved_parents,
            unresolved_ranges=schema.unresolved_ranges,
            unresolved_domains=schema.unresolved_domains,
            absent=schema.absent,
            closed_ranges=schema.closed_ranges,
        ),
    )


async def _frozen_schema(
    deps: GraphDeps, *, code_index: CodeIndex, deadline: float
) -> FrozenSchema:
    """The built bodies for `code_index` from `deps.schema`, building them on a miss and
    waiting until `deadline`. Refresh the generation first, as the documentation route does, so
    the stamp is fresh after an idle gap.

    Raises:
        SchemaReadFailed: the build failed, or the entry keeps its refusal.
    """
    await deps.generation.refresh_if_stale(timeout_s=SCHEMA_GENERATION_WAIT_SECONDS)
    token = deps.generation.stamp().token
    catalogue = current_catalogue()
    key = SchemaKey(
        generation=token if token is not None else UNKNOWN_KEY,
        code_index=code_index.stamp.content_hash,
        origin_catalogue=catalogue.build_id,
    )

    def build(_keys: list[SchemaKey], flight_deadline: float) -> dict[SchemaKey, SchemaEntry]:
        entry = _build_entry(
            deps,
            code_index=code_index,
            catalogue=catalogue,
            generation=token,
            deadline=flight_deadline,
        )
        return {key: entry}

    try:
        found = await deps.schema.aget_or_fetch_many(
            [key], log_queue_wait(build), deadline=deadline, value_ttl=_entry_ttl
        )
    except WaitTimeout:
        raise SchemaBuildPending() from None
    except TimeoutError as exc:
        raise SchemaBuildExpired() from exc
    except GraphReadError as exc:
        raise SchemaReadFailed(exc) from exc
    entry = found[key]
    if isinstance(entry, RefusedSchema):
        raise SchemaReadFailed(entry.error())
    return entry


def classify_schema_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify the build's waits and read failures, which this door raises as its own types,
    and the check's concurrency limit. The two retryable waits carry `cause.code`
    `schema_building` or `check_busy`, so a caller can tell a build in progress from a full
    check pool."""
    if isinstance(exc, SchemaBuildPending | SchemaCheckBusy):
        busy = isinstance(exc, SchemaCheckBusy)
        retry_after = CHECK_BUSY_RETRY_AFTER_SECONDS if busy else SCHEMA_RETRY_AFTER_SECONDS
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=str(exc),
            retryable=True,
            cause=Cause(code="check_busy" if busy else "schema_building"),
            headers={"Retry-After": str(retry_after)},
        )
    if isinstance(exc, UnOriginNotServed | OriginUsageNotServed | NodeNotHeldError):
        return ProblemSpec(
            status=404,
            code="not_found",
            detail=str(exc),
            retryable=False,
            cause=Cause(
                code="node_not_held" if isinstance(exc, NodeNotHeldError) else "origin_not_served"
            ),
        )
    if isinstance(exc, NodeViewTooLarge):
        return ProblemSpec(
            status=500,
            code=SERVER_ERROR_FALLBACK_CODE,
            detail=str(exc),
            retryable=False,
            cause=Cause(code="node_view_too_large"),
            log=f"Graph schema door refused a node view past its byte cap: {exc}",
        )
    if isinstance(exc, NodeViewNotAdvancing):
        return ProblemSpec(
            status=500,
            code=SERVER_ERROR_FALLBACK_CODE,
            detail=str(exc),
            retryable=False,
            cause=Cause(code="graph_not_advancing"),
            log=f"Graph schema door refused a node view whose paged read did not advance: {exc}",
        )
    if isinstance(exc, NodeViewTimedOut):
        return ProblemSpec(
            status=TIMEOUT_DEFAULT.status,
            code=TIMEOUT_DEFAULT.code,
            detail=str(exc),
            retryable=TIMEOUT_DEFAULT.retryable,
        )
    if isinstance(exc, SchemaBuildExpired):
        return ProblemSpec(
            status=TIMEOUT_DEFAULT.status,
            code=TIMEOUT_DEFAULT.code,
            detail="the schema build passed its deadline; retry",
            retryable=TIMEOUT_DEFAULT.retryable,
            log="Graph schema door's build passed its flight deadline",
        )
    if not isinstance(exc, SchemaReadFailed):
        return None
    error = exc.error
    if isinstance(error, UnTermsRateLimitedError | UnTermsListingEmptyError):
        rate_limited = isinstance(error, UnTermsRateLimitedError)
        retry_after = (
            error.retry_after if error.retry_after is not None else SCHEMA_REFUSAL_TTL_SECONDS
        )
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=str(error),
            retryable=True,
            cause=Cause(code="upstream_rate_limited" if rate_limited else "listing_incomplete"),
            headers={"Retry-After": str(math.ceil(retry_after))},
            log=f"Graph schema door refused a UN terms build: {error}",
        )
    if isinstance(error, SchemaListingTruncatedError):
        return ProblemSpec(
            status=500,
            code=SERVER_ERROR_FALLBACK_CODE,
            detail=str(error),
            retryable=False,
            log=f"Graph schema door refused a truncated term listing: {error}",
        )
    if isinstance(error, SchemaNoDfoDeclarationError):
        return ProblemSpec(
            status=500,
            code=SERVER_ERROR_FALLBACK_CODE,
            detail=str(error),
            retryable=False,
            log=f"Graph schema door refused a document no DFO provenance declares: {error}",
        )
    if isinstance(error, GraphDeadlineExceededError):
        return ProblemSpec(
            status=TIMEOUT_DEFAULT.status,
            code=TIMEOUT_DEFAULT.code,
            detail="the deadline expired during the schema door's graph reads; retry",
            retryable=TIMEOUT_DEFAULT.retryable,
        )
    if error.upstream_status == _RATE_LIMITED:
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the graph is limiting the rate of reads; retry",
            retryable=True,
            headers={"Retry-After": str(GRAPH_RATE_LIMITED_RETRY_AFTER_SECONDS)},
            log=f"Graph schema door's DCP read still answered 429: {error}",
        )
    return graph_read_problem(error)._replace(
        log=f"Graph schema door's DCP read failed: {error}", log_traceback=True
    )


def _freeze_usage(
    reader: GraphNodeReader,
    *,
    code_index: CodeIndex,
    coverage: ProvenanceCoverage,
    deadline: float,
) -> FrozenJSON:
    """Read every Provenance node of `code_index` in one batch, confirming each empty answer
    alone, and freeze the usage body built from it."""
    nodes = reader.node_arcs_many(
        sorted(code_index.variables_by_provenance),
        properties=PROVENANCE_NODE_PROPERTIES,
        individual_probe_budget=None,
        deadline=deadline,
    )
    return _freeze(build_usage(code_index=code_index, coverage=coverage, nodes=nodes))


async def _frozen_usage(deps: GraphDeps) -> FrozenJSON:
    """The usage body of the bound code index and the served coverage build, from
    `deps.dataset_usage`. Refresh the generation first, as `_frozen_schema` does, so
    the cache's stamp is fresh after an idle gap. A miss reads the Provenance nodes, builds and
    encodes the body on the cache's executor, and concurrent misses share that one build. A
    request waits until `GRAPH_NODE_TIMEOUT_SECONDS` from its start.

    Raises:
        SchemaBuildPending: the request's wait passed while the build runs on.
        SchemaBuildExpired: the build outlived its flight deadline.
        SchemaReadFailed: the Provenance node read failed.
    """
    await deps.generation.refresh_if_stale()
    code_index = get_index()
    snapshot = deps.coverage.snapshot()
    key = UsageKey(
        code_index=code_index.stamp.content_hash, coverage_build=snapshot.manifest.build_id
    )

    def build(_keys: list[UsageKey], flight_deadline: float) -> dict[UsageKey, FrozenJSON]:
        frozen = _freeze_usage(
            deps.graph, code_index=code_index, coverage=snapshot.index, deadline=flight_deadline
        )
        return {key: frozen}

    try:
        found = await deps.dataset_usage.aget_or_fetch_many(
            [key], build, deadline=_request_deadline()
        )
    except WaitTimeout:
        raise SchemaBuildPending() from None
    except TimeoutError as exc:
        raise SchemaBuildExpired() from exc
    except GraphReadError as exc:
        raise SchemaReadFailed(exc) from exc
    return found[key]


async def _frozen_un_terms(deps: GraphDeps, *, deadline: float | None = None) -> FrozenJSON:
    """The UN terms body from `deps.un_terms`, building it on a miss. With no body served yet,
    a request waits until `deadline`, `GRAPH_NODE_TIMEOUT_SECONDS` from its start when `None`.
    Once one is served, a miss starts or joins the build and answers that previous body at
    once, and so does a failed or refused build, so a new UN generation or catalogue build
    shows when its build lands.

    Raises:
        UnOriginNotServed: the bound catalogue does not serve the UN origin, or the surface has
            no UN reader. Nothing is read.
        SchemaBuildPending: the request's wait passed while the build runs on.
        SchemaBuildExpired: the build outlived its flight deadline.
        SchemaReadFailed: a listing failed, or the entry keeps the build's refusal.
    """
    catalogue = current_catalogue()
    un_terms = deps.un_terms
    if un_terms is None or not catalogue.un_system_enabled:
        raise UnOriginNotServed()

    def build(_keys: list[str], flight_deadline: float) -> dict[str, UnTermsEntry]:
        try:
            terms = build_un_terms(
                reader=un_terms.reader,
                catalogue=catalogue,
                deadline=flight_deadline,
                stop=un_terms.stop,
            )
        except SchemaListingTruncatedError as exc:
            refusal = RefusedSchema(error=partial(SchemaListingTruncatedError, str(exc)))
        except UnTermsListingEmptyError as exc:
            refusal = _retry_refusal(
                partial(UnTermsListingEmptyError, exc.kind), ttl_s=SCHEMA_REFUSAL_TTL_SECONDS
            )
        except UnTermsRateLimitedError as exc:
            refusal = _retry_refusal(
                partial(UnTermsRateLimitedError, str(exc)),
                ttl_s=max(SCHEMA_REFUSAL_TTL_SECONDS, exc.retry_after or 0.0),
            )
        else:
            return {UN_TERMS_KEY: _freeze(terms)}
        return {UN_TERMS_KEY: refusal}

    previous = un_terms.served
    if previous is not None:
        wait_until = time.monotonic()
    else:
        wait_until = deadline if deadline is not None else _request_deadline()
    try:
        found = await un_terms.cache.aget_or_fetch_many(
            [UN_TERMS_KEY], build, deadline=wait_until, value_ttl=_entry_ttl
        )
    except WaitTimeout:
        if previous is not None:
            return previous
        raise SchemaBuildPending() from None
    except (TimeoutError, GraphReadError) as exc:
        if previous is not None:
            logger.warning("UN terms build failed, serving the previous body: %s", exc)
            return previous
        if isinstance(exc, GraphReadError):
            raise SchemaReadFailed(exc) from exc
        raise SchemaBuildExpired() from exc
    entry = found[UN_TERMS_KEY]
    if isinstance(entry, RefusedSchema):
        if previous is not None:
            return previous
        raise SchemaReadFailed(entry.error())
    un_terms.served = entry
    return entry


def _retry_refusal(error: Callable[[], GraphReadError], *, ttl_s: float) -> RefusedSchema:
    """A refusal kept for `ttl_s` whose error names, as `retry_after`, the seconds left until
    the entry expires and a request builds again, one at least."""
    retry_at = time.monotonic() + ttl_s

    def make() -> GraphReadError:
        made = error()
        made.retry_after = max(1.0, retry_at - time.monotonic())
        return made

    return RefusedSchema(error=make, ttl_s=ttl_s)


async def warm_un_terms(deps: GraphDeps, *, timeout_s: float) -> None:
    """Build the entry `GET /api/graph/schema/terms/un_system` would build now, under the bound
    origin catalogue, waiting at most `timeout_s`, so the route has a body to answer while a
    later UN generation's build runs. A surface that does not serve the UN origin builds
    nothing. A failure or an unfinished build logs and returns, and the first request waits for
    the build as on a cold entry."""
    if deps.un_terms is None or not current_catalogue().un_system_enabled:
        return
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            await _frozen_un_terms(deps, deadline=started + timeout_s)
    except (TimeoutError, SchemaBuildPending):
        logger.warning(
            "UN terms warm-up did not finish within %.0f s; the first request waits for it",
            timeout_s,
        )
    except Exception:
        logger.warning("UN terms warm-up failed", exc_info=True)
    else:
        logger.info("UN terms warm-up built in %.2f s", time.monotonic() - started)


def _freeze_origin_usage(
    catalogue: CatalogueSnapshot, key: OriginUsageKey, *, deadline: float
) -> FrozenJSON:
    """The frozen usage body of `key.origin` in `catalogue` under the bound release, logging
    the build's time and size.

    Raises:
        TimeoutError: `deadline` passed before the encoding.
    """
    started = time.monotonic()
    usage = build_origin_usage(catalogue, key.origin)
    built = time.monotonic()
    if built >= deadline:
        raise TimeoutError(f"the {key.origin} usage build outlived its flight deadline")
    frozen = _freeze(usage)
    ended = time.monotonic()
    logger.info(
        "origin usage %s under catalogue %s and release %s took %d ms: build %d ms; "
        "encode %d ms %d bytes",
        key.origin,
        catalogue.build_id,
        key.release,
        round((ended - started) * 1_000),
        round((built - started) * 1_000),
        round((ended - built) * 1_000),
        len(frozen.identity),
    )
    return frozen


async def _frozen_origin_usage(deps: GraphDeps, origin: UsageOrigin) -> FrozenJSON:
    """The usage body of `origin` from `deps.origin_usage`, building it on the cache's executor
    on a miss, with concurrent misses sharing that one build. The cache keys on `origin` and the
    bound release's id under the bound catalogue's build (`CatalogueGeneration`), and the build
    runs under the context of the request that starts it, so it reads that release. The build
    ignores the request's source scope. A request waits until `GRAPH_NODE_TIMEOUT_SECONDS` from
    its start.

    Raises:
        UnOriginNotServed: `origin` is the UN System and the bound catalogue does not serve it.
            Nothing is built.
        OriginUsageNotServed: the bound catalogue is not a loaded `CatalogueSnapshot`, whose
            tables the build reads and the `Catalogue` protocol leaves out. Nothing is built.
        SchemaBuildPending: the request's wait passed while the build runs on.
        SchemaBuildExpired: the build outlived its flight deadline.
    """
    catalogue = current_catalogue()
    if origin == "un_system" and not catalogue.un_system_enabled:
        raise UnOriginNotServed()
    if not isinstance(catalogue, CatalogueSnapshot):
        raise OriginUsageNotServed(origin)
    key = OriginUsageKey(origin=origin, release=served_origin_key())

    def build(
        _keys: list[OriginUsageKey], flight_deadline: float
    ) -> dict[OriginUsageKey, FrozenJSON]:
        return {key: _freeze_origin_usage(catalogue, key, deadline=flight_deadline)}

    try:
        found = await deps.origin_usage.aget_or_fetch_many(
            [key], build, deadline=_request_deadline()
        )
    except WaitTimeout:
        raise SchemaBuildPending() from None
    except TimeoutError as exc:
        raise SchemaBuildExpired() from exc
    return found[key]


@dataclass(frozen=True, slots=True)
class _ViewSide:
    """One deployment's reader, view cache and miss cache."""

    reader: NodeViewReader
    cache: SingleFlightCache[NodeViewKey, NodeViewEntry]
    misses: SingleFlightCache[NodeViewKey, NodeViewMiss]


def _un_side(views: NodeViews, catalogue: Catalogue) -> _ViewSide | None:
    """The UN deployment's reader and caches, or `None` when the bound catalogue does not serve
    the UN origin or the surface has no UN reader."""
    if (
        views.un_reader is None
        or views.un is None
        or views.un_misses is None
        or not catalogue.un_system_enabled
    ):
        return None
    return _ViewSide(reader=views.un_reader, cache=views.un, misses=views.un_misses)


def _refuse(miss: NodeViewMiss, dcid: str) -> Exception:
    if miss is NodeViewMiss.TOO_LARGE:
        return NodeViewTooLarge(dcid)
    if miss is NodeViewMiss.NOT_ADVANCING:
        return NodeViewNotAdvancing(dcid)
    return NodeNotHeldError(dcid)


def _not_advancing(exc: BaseException) -> bool:
    """Whether `exc` or an exception in its `__cause__` chain is a
    `dc_graph.GraphNotAdvancingError`."""
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, GraphNotAdvancingError):
            return True
        cause = cause.__cause__
    return False


async def _node_view_from(
    views: NodeViews,
    side: _ViewSide,
    *,
    key: NodeViewKey,
    catalogue: Catalogue,
    generation: str | None,
    deadline: float,
) -> FrozenJSON:
    """The view of `key.dcid` read from `side.reader`, from `side.cache`, building it on a miss
    with its reads on `views.reads` and waiting until `deadline`. A refusal is kept in
    `side.misses`, which answers it again with no read.

    Raises:
        NodeNotHeldError: `side.reader` holds no such node.
        NodeViewTooLarge: a read answered past the view's byte cap.
        NodeViewNotAdvancing: a paged read did not advance.
        NodeViewTimedOut: the wait or the flight passed its deadline.
        SchemaReadFailed: a read failed.
        DcpExecutorSaturated: every flight worker is busy.
    """
    if (miss := side.misses.cached_many([key]).get(key)) is not None:
        raise _refuse(miss, key.dcid)
    stamp_key = side.misses.stamp_key()

    def build(_keys: list[NodeViewKey], flight_deadline: float) -> dict[NodeViewKey, NodeViewEntry]:
        try:
            view = build_node_view(
                key.dcid,
                reader=side.reader,
                catalogue=catalogue,
                generation=generation,
                executor=views.reads,
                deadline=flight_deadline,
            )
        except NodeNotHeldError:
            return {key: NodeViewMiss.NOT_HELD}
        except GraphCapExceededError:
            return {key: NodeViewMiss.TOO_LARGE}
        except GraphReadError as exc:
            if _not_advancing(exc):
                logger.info("node view of %r: a paged read did not advance: %s", key.dcid, exc)
                return {key: NodeViewMiss.NOT_ADVANCING}
            raise
        return {key: _freeze(view)}

    try:
        found = await side.cache.aget_or_fetch_many(
            [key],
            log_queue_wait(build),
            deadline=deadline,
            is_cacheable=lambda entry: not isinstance(entry, NodeViewMiss),
        )
    except TimeoutError as exc:
        raise NodeViewTimedOut() from exc
    except GraphReadError as exc:
        raise SchemaReadFailed(exc) from exc
    entry = found[key]
    if isinstance(entry, NodeViewMiss):
        side.misses.offer(key, entry, stamp_key=stamp_key)
        raise _refuse(entry, key.dcid)
    return entry


async def _frozen_node_view(deps: GraphDeps, dcid: str) -> FrozenJSON:
    """The view of `dcid`. A UN id is read from the UN deployment, once the bound catalogue
    serves the UN origin. Any other id is read from the mixer, and from the UN deployment when
    the mixer holds no such node and the catalogue serves the UN origin. A request waits until
    `GRAPH_NODE_TIMEOUT_SECONDS` from its start over both reads, and each read's flight has a
    deadline of its own.

    Raises:
        DcidTooLong: `dcid` is longer than `DCID_MAX_LENGTH`.
        GraphDepsNotSet: the surface has no node view reader.
        UnOriginNotServed: `dcid` is a UN id and the UN origin is not served. Nothing is read.
        NodeNotHeldError: no deployment read holds the node.
        NodeViewTimedOut: a wait or a flight passed its deadline.
        SchemaReadFailed: a read failed.
        DcpExecutorSaturated: every flight worker is busy.
    """
    if len(dcid) > DCID_MAX_LENGTH:
        raise DcidTooLong(len(dcid))
    views = deps.node_views
    if views is None:
        raise GraphDepsNotSet()
    deadline = _request_deadline()
    catalogue = current_catalogue()
    key = NodeViewKey(
        dcid=dcid,
        origin_catalogue=catalogue.build_id,
        un_system_enabled=catalogue.un_system_enabled,
    )
    un = _un_side(views, catalogue)
    if not is_un_dcid(dcid):
        await deps.generation.refresh_if_stale()
        mixer = _ViewSide(reader=views.mixer, cache=views.base, misses=views.base_misses)
        try:
            return await _node_view_from(
                views,
                mixer,
                key=key,
                catalogue=catalogue,
                generation=deps.generation.stamp().token,
                deadline=deadline,
            )
        except NodeNotHeldError:
            if un is None:
                raise
    elif un is None:
        raise UnOriginNotServed()
    return await _node_view_from(
        views, un, key=key, catalogue=catalogue, generation=None, deadline=deadline
    )


async def warm_schema(deps: GraphDeps, *, timeout_s: float) -> None:
    """Build the entry `GET /api/graph/schema` would build now, under the bound origin
    catalogue and served release, waiting at most `timeout_s`. A build still running then
    runs on under its flight deadline and lands for the request that joins it. A failure or
    an unfinished build logs and returns. The first request after a failure builds again,
    unless the entry kept the build's refusal (`RefusedSchema`), which a request within
    `SCHEMA_REFUSAL_TTL_SECONDS` (60 s) answers with no build."""
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            await _frozen_schema(deps, code_index=get_index(), deadline=started + timeout_s)
    except (TimeoutError, SchemaBuildPending):
        logger.warning(
            "Graph schema warm-up did not finish within %.0f s; the first request waits for it",
            timeout_s,
        )
    except Exception:
        logger.warning("Graph schema warm-up failed", exc_info=True)
    else:
        logger.info("Graph schema warm-up built in %.2f s", time.monotonic() - started)


async def get_schema(deps: GraphDepsDep, request: Request) -> Response:
    frozen = await _frozen_schema(deps, code_index=get_index(), deadline=_request_deadline())
    return frozen.document.respond(request)


async def get_schema_terms(deps: GraphDepsDep, request: Request) -> Response:
    frozen = await _frozen_schema(deps, code_index=get_index(), deadline=_request_deadline())
    return frozen.terms.respond(request)


async def get_un_schema_terms(deps: GraphDepsDep, request: Request) -> Response:
    return (await _frozen_un_terms(deps)).respond(request)


def _check_body(
    rows: Sequence[SchemaCheckRow],
    *,
    reads: frozenset[str],
    lookup: SchemaLookup,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    un_served: bool,
    generation: str | None,
    deadline: float,
) -> bytes:
    """The encoded response of one check of `rows`, the first rows of the request, from
    `check_rows` and `SchemaCheckResponse`'s JSON."""
    checked = check_rows(
        rows,
        reads=reads,
        lookup=lookup,
        reader=reader,
        code_index=code_index,
        un_served=un_served,
        generation=generation,
        deadline=deadline,
    )
    response = SchemaCheckResponse(
        generation=generation, rows_checked=checked.rows_checked, findings=checked.findings
    )
    return response.model_dump_json().encode()


async def post_schema_check(deps: GraphDepsDep, body: SchemaCheckRequest) -> Response:
    deadline = _request_deadline(GRAPH_SCHEMA_CHECK_TIMEOUT_SECONDS)
    code_index = get_index()
    frozen = await _frozen_schema(deps, code_index=code_index, deadline=deadline)
    un_served = current_catalogue().un_system_enabled
    plan = plan_reads(body.rows, lookup=frozen.lookup, un_served=un_served)
    try:
        async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
            encoded = await run_dcp_read(
                deps.check_executor,
                _check_body,
                body.rows[: plan.row_count],
                reads=plan.ids,
                lookup=frozen.lookup,
                reader=deps.graph,
                code_index=code_index,
                un_served=un_served,
                generation=frozen.generation,
                deadline=deadline,
            )
    except DcpExecutorSaturated:
        raise SchemaCheckBusy() from None
    except GraphReadError as exc:
        raise SchemaReadFailed(exc) from exc
    except TimeoutError:
        error = GraphDeadlineExceededError("the schema check passed its deadline")
        raise SchemaReadFailed(error) from None
    return Response(content=encoded, media_type="application/json")


async def get_schema_node(dcid: str, deps: GraphDepsDep, request: Request) -> Response:
    return (await _frozen_node_view(deps, dcid)).respond(request)


async def get_schema_usage(deps: GraphDepsDep, request: Request) -> Response:
    return (await _frozen_usage(deps)).respond(request)


async def get_schema_usage_base(deps: GraphDepsDep, request: Request) -> Response:
    return (await _frozen_origin_usage(deps, "base")).respond(request)


async def get_schema_usage_un_system(deps: GraphDepsDep, request: Request) -> Response:
    return (await _frozen_origin_usage(deps, "un_system")).respond(request)


def build_schema_door() -> DoorContribution:
    router = APIRouter()
    # `response_model` documents the response body in OpenAPI. The handlers' responses pass
    # through.
    router.add_api_route(
        "/api/graph/schema", get_schema, methods=["GET"], response_model=SchemaDocument
    )
    router.add_api_route(
        "/api/graph/schema/terms",
        get_schema_terms,
        methods=["GET"],
        response_model=SchemaTerms,
    )
    router.add_api_route(
        "/api/graph/schema/terms/un_system",
        get_un_schema_terms,
        methods=["GET"],
        response_model=SchemaTerms,
    )
    router.add_api_route(
        "/api/graph/schema/usage",
        get_schema_usage,
        methods=["GET"],
        response_model=DatasetUsage,
    )
    router.add_api_route(
        "/api/graph/schema/usage/base",
        get_schema_usage_base,
        methods=["GET"],
        response_model=DatasetUsage,
    )
    router.add_api_route(
        "/api/graph/schema/usage/un_system",
        get_schema_usage_un_system,
        methods=["GET"],
        response_model=DatasetUsage,
    )
    router.add_api_route(
        "/api/graph/schema/node/{dcid:dcid}",
        get_schema_node,
        methods=["GET"],
        response_model=SchemaNodeView,
    )
    router.add_api_route(
        "/api/graph/schema/check",
        post_schema_check,
        methods=["POST"],
        response_model=SchemaCheckResponse,
    )
    return DoorContribution(
        router=router,
        classify=classify_schema_exception,
        exception_types=(
            SchemaBuildPending,
            SchemaBuildExpired,
            SchemaReadFailed,
            SchemaCheckBusy,
            UnOriginNotServed,
            NodeNotHeldError,
            NodeViewTooLarge,
            NodeViewNotAdvancing,
            NodeViewTimedOut,
        ),
        exception_exemplars=(
            UnOriginNotServed(),
            NodeNotHeldError("exemplar"),
            NodeViewTooLarge("exemplar"),
            NodeViewNotAdvancing("exemplar"),
            NodeViewTimedOut(),
            SchemaCheckBusy(),
            SchemaBuildPending(),
            SchemaBuildExpired(),
            SchemaReadFailed(SchemaListingTruncatedError("exemplar")),
            SchemaReadFailed(SchemaNoDfoDeclarationError(0)),
            SchemaReadFailed(UnTermsRateLimitedError("exemplar")),
            SchemaReadFailed(UnTermsListingEmptyError("Property")),
            SchemaReadFailed(GraphDeadlineExceededError("exemplar")),
            SchemaReadFailed(GraphReadError("exemplar", upstream_status=400)),
            SchemaReadFailed(GraphReadError("exemplar", upstream_status=429)),
            SchemaReadFailed(GraphReadError("exemplar", upstream_status=503)),
        ),
    )
