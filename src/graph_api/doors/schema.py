"""Serve the deployment's schema document and the list of the DFO graph's Classes and
Properties.

`GET /api/graph/schema` serves `SchemaDocument` and `GET /api/graph/schema/terms`
serves `SchemaTerms`, every Class and Property the mixer lists by `typeOf` and none of the UN
deployment's. One build (`graph_api.schema_document.build_schema`) yields both,
and `GraphDeps.schema` holds its result in a single-flight cache of two entries, keyed by
the graph generation token, the content hash of the code index the build read and the build of
the origin catalogue whose provenance table names origins for `declared_by`, beside the cache's
own generation stamp. Two entries let a request stamped with an unknown generation
keep the entry of the known one. The lifespan registers it as `graph_api.schema`, so a
generation change clears it.

A request refreshes a stale generation stamp, then waits for the entry until
`GRAPH_NODE_TIMEOUT_SECONDS` from its start. A miss runs the whole build as one job on the
DCP-read executor, holding one worker, under a flight deadline of
`SCHEMA_FLIGHT_TIMEOUT_SECONDS` from submission. Concurrent requests share the flight, and a
request whose own deadline passes first answers retryable 503 `service_unavailable` with
`Retry-After: 5` and `cause: {"code": "schema_building"}` while the flight runs on and stores
its result for the next request. A flight
that outlives its deadline stores nothing, and its waiters answer 504 `timeout`. An executor
with every worker busy refuses the build with `DcpExecutorSaturated`, which the surface answers
as retryable 503. A read past its deadline answers 504 `timeout`. A listing that reached its
cap, or a document with no node DFO declares, is a server fault a retry does not clear and
answers non-retryable 500 `internal_error`. A DCP 4xx answers 502 `upstream_error` and any other
read failure retryable 503. Under a known generation the two refusals cannot change while the
generation holds, so the entry keeps them as `RefusedSchema` and every request for the key
answers the same 500 with no build. Under an unknown generation nothing would clear a kept
refusal, so the build raises it unkept and the next request builds again. Any other failed
build is not kept.

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
`CHECK_MAX_READ_IDS`. Whether the bound catalogue serves the UN origin decides whether a UN id
is read or gets `origin_not_served`.

Every check runs as one job on `GraphDeps.check_executor`, whose
`GRAPH_SCHEMA_CHECK_CONCURRENCY` (2) workers are its own, so a check takes none of the
shared DCP-read workers. The job reads, checks and encodes the response, and the event loop
only wraps its bytes. A worker stays held until its job returns, even after its request is
cancelled or times out, so at most two checks run at once in the process. One more is refused at
submit and answers retryable 503 `service_unavailable` with `Retry-After: 5` and
`cause: {"code": "check_busy"}` (`SchemaCheckBusy`). A read past the check's deadline or a failed read answers as the GET
routes answer it, with no findings. The request stops waiting at the check's deadline and
answers 504 `timeout` while its job runs on, so a read that answers late yields no 200. The job
stops before the row whose findings would pass `CHECK_MAX_FINDINGS`, or take the encoded
response past `CHECK_MAX_BODY_BYTES` (4 MB), and always takes the first row. A caller resumes at
`rows_checked` when it is less than the rows it sent.

`GET /api/graph/schema/usage` serves `DatasetUsage`, built by
`graph_api.dataset_usage.build_usage` from the bound code index and the served coverage index
with no graph read. `GraphDeps.dataset_usage`, a single-flight cache of one entry keyed by the
code index's content hash and the coverage build (`CoverageManifest.build_id`), holds the
frozen body, and the lifespan registers it as `graph_api.dataset_usage`, so a generation change
clears it too. A miss builds the body
on the cache's own one-worker executor, and concurrent misses share that build. A request whose
wait passes `GRAPH_NODE_TIMEOUT_SECONDS` answers retryable 503 with `Retry-After: 5` and
`cause: {"code": "schema_building"}`, and a build
past its flight deadline answers 504, as for the other two routes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

from fastapi import APIRouter, Request, Response

from code_index import CodeIndex
from code_index.shared import get_index
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
from graph_api.dataset_usage import ProvenanceCoverage, build_usage
from graph_api.deps import (
    DcpExecutorSaturated,
    GraphDeps,
    GraphDepsDep,
    log_queue_wait,
    run_dcp_read,
)
from graph_api.doors.node import graph_read_problem
from graph_api.reader import GraphNodeReader
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
from graph_generation import GenerationProvider
from graph_generation.stamp import UNKNOWN_KEY
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from origin_catalogue import Catalogue, current_catalogue

SCHEMA_RETRY_AFTER_SECONDS = 5
"""The `Retry-After` a request sends when its deadline passes while the build runs on."""

_MAX_AGE_SECONDS = 300


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
    """A build refusal that cannot change while the generation holds: a truncated term listing,
    or a document with no node DFO declares. `error` makes a fresh exception for each
    request it answers."""

    error: Callable[[], GraphReadError]


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


class UsageKey(NamedTuple):
    """The content hash of the code index and the coverage build the usage body reads."""

    code_index: str
    coverage_build: str


def new_schema_cache[K, V](
    *, entries: int, executor: Executor, generation: GenerationProvider
) -> SingleFlightCache[K, V]:
    """A cache of this door's bodies holding `entries` entries, each built on `executor` within
    `SCHEMA_FLIGHT_TIMEOUT_SECONDS`, with a key that carries `generation`'s stamp.
    `GraphDeps.schema` holds two, so a request stamped with an unknown generation keeps the
    entry of the known one, and `GraphDeps.dataset_usage` holds one."""
    return SingleFlightCache(
        maxsize=entries,
        flight_timeout_s=SCHEMA_FLIGHT_TIMEOUT_SECONDS,
        executor=executor,
        generation=generation,
    )


def _freeze(model: SchemaDocument | SchemaTerms | DatasetUsage) -> FrozenJSON:
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
    """Build and freeze the bodies, or keep a refusal that cannot change within the
    generation. Under an unknown generation, `generation` `None`, nothing would clear a kept
    refusal before the stamp's age runs out, so the refusal is raised and not kept."""
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
    return FrozenSchema(
        document=_freeze(schema.document),
        terms=_freeze(schema.terms),
        generation=schema.document.generation,
        lookup=SchemaLookup.of(schema.document),
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
    await deps.generation.refresh_if_stale()
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
            [key], log_queue_wait(build), deadline=deadline
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
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=str(exc),
            retryable=True,
            cause=Cause(
                code="check_busy" if isinstance(exc, SchemaCheckBusy) else "schema_building"
            ),
            headers={"Retry-After": str(SCHEMA_RETRY_AFTER_SECONDS)},
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
    return graph_read_problem(error)._replace(
        log=f"Graph schema door's DCP read failed: {error}", log_traceback=True
    )


def _freeze_usage(code_index: CodeIndex, coverage: ProvenanceCoverage) -> FrozenJSON:
    return _freeze(build_usage(code_index=code_index, coverage=coverage))


async def _frozen_usage(deps: GraphDeps) -> FrozenJSON:
    """The usage body of the bound code index and the served coverage build, from
    `deps.dataset_usage`. Refresh the generation first, as `_frozen_schema` does, so
    the cache's stamp is fresh after an idle gap. A miss builds and encodes the body on the cache's executor,
    and concurrent misses share that one build. A request waits until `GRAPH_NODE_TIMEOUT_SECONDS`
    from its start.

    Raises:
        SchemaBuildPending: the request's wait passed while the build runs on.
        SchemaBuildExpired: the build outlived its flight deadline.
    """
    await deps.generation.refresh_if_stale()
    code_index = get_index()
    snapshot = deps.coverage.snapshot()
    key = UsageKey(
        code_index=code_index.stamp.content_hash, coverage_build=snapshot.manifest.build_id
    )

    def build(_keys: list[UsageKey], _flight_deadline: float) -> dict[UsageKey, FrozenJSON]:
        return {key: _freeze_usage(code_index, snapshot.index)}

    try:
        found = await deps.dataset_usage.aget_or_fetch_many(
            [key], build, deadline=_request_deadline()
        )
    except WaitTimeout:
        raise SchemaBuildPending() from None
    except TimeoutError as exc:
        raise SchemaBuildExpired() from exc
    return found[key]


async def get_schema(deps: GraphDepsDep, request: Request) -> Response:
    frozen = await _frozen_schema(deps, code_index=get_index(), deadline=_request_deadline())
    return frozen.document.respond(request)


async def get_schema_terms(deps: GraphDepsDep, request: Request) -> Response:
    frozen = await _frozen_schema(deps, code_index=get_index(), deadline=_request_deadline())
    return frozen.terms.respond(request)


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


async def get_schema_usage(deps: GraphDepsDep, request: Request) -> Response:
    return (await _frozen_usage(deps)).respond(request)


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
        "/api/graph/schema/usage",
        get_schema_usage,
        methods=["GET"],
        response_model=DatasetUsage,
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
        ),
        exception_exemplars=(
            SchemaCheckBusy(),
            SchemaBuildPending(),
            SchemaBuildExpired(),
            SchemaReadFailed(SchemaListingTruncatedError("exemplar")),
            SchemaReadFailed(SchemaNoDfoDeclarationError(0)),
            SchemaReadFailed(GraphDeadlineExceededError("exemplar")),
            SchemaReadFailed(GraphReadError("exemplar", upstream_status=400)),
            SchemaReadFailed(GraphReadError("exemplar", upstream_status=503)),
        ),
    )
