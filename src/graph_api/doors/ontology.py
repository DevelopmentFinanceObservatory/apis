"""Serve the instance's ontology document and the list of every Class and Property.

`GET /api/graph/ontology` serves `OntologyDocument` and `GET /api/graph/ontology/terms`
serves `OntologyTerms`. One build (`graph_api.ontology_document.build_ontology`) yields both,
and `GraphDeps.ontology` holds its result in a single-flight cache of one entry, keyed by the
graph generation token and the content hash of the code index the build read, beside the
cache's own generation stamp. The lifespan registers it as `graph_api.ontology`, so a
generation change clears it.

A request refreshes a stale generation stamp, then waits for the entry until
`GRAPH_NODE_TIMEOUT_SECONDS` from its start. A miss runs the whole build as one job on the
DCP-read executor, holding one worker, under a flight deadline of
`ONTOLOGY_FLIGHT_TIMEOUT_SECONDS` from submission. Concurrent requests share the flight, and a
request whose own deadline passes first answers retryable 503 `service_unavailable` with
`Retry-After: 5` while the flight runs on and stores its result for the next request. A flight
that outlives its deadline stores nothing, and its waiters answer 504 `timeout`. An executor
with every worker busy refuses the build with `DcpExecutorSaturated`, which the surface answers
as retryable 503. A read past its deadline answers 504 `timeout`, a listing that reached its
cap non-retryable 503 `service_unavailable`, a DCP 4xx 502 `upstream_error` and any other read
failure retryable 503. The non-retryable refusal cannot change while the generation holds, so
the entry keeps it as `RefusedOntology` and every request for the key answers the same 503 with
no build. Any other failed build is not kept.

The cached value holds each body encoded once in the models' field order, so two builds of one
graph give the same bytes. Each answers `public, max-age=300` with a strong `ETag` over those
bytes and a gzip representation, and answers `If-None-Match` with 304
(`dfo_platform.static_response.FrozenJSON`).

`POST /api/graph/ontology/check` takes 1 to `CHECK_MAX_ROWS` candidate StatVar definitions and
answers every fault `graph_api.ontology_check.check_rows` finds in them, with the generation
of the build it checked against. The entry keeps, beside its frozen bodies, the build's
`OntologyLookup`, built once per entry. The check compares served definitions against the code
index the request computed the entry's key from, which is the index the build read. The
check's deadline is `GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS` from its start. It waits on the same
flight as the GET routes until that deadline and answers their 503 and 504 the same way, so the
first check after a graph change builds the document before its caps are decided.

A body outside the request model is refused 400 `request_invalid`: `rows` empty or longer than
`CHECK_MAX_ROWS`, a row past `CHECK_MAX_CONSTRAINT_PROPERTIES` or
`CHECK_MAX_OBSERVATION_PROPERTIES`, a `ref` or a constraint value string past 512 characters,
or a constraint value that is a number past plus or minus 2**53, `NaN` or an infinity. With the
document in hand, `plan_reads` decides on the event loop, before anything is submitted, which
ids the check reads: more than `CHECK_MAX_READ_IDS`, or a UN id while the bound catalogue does
not serve the UN origin, is refused 400 `request_invalid` with pointer `/rows`.

Every check runs as one job on `GraphDeps.check_executor`, whose
`GRAPH_ONTOLOGY_CHECK_CONCURRENCY` (2) workers are its own, so a check takes none of the
shared DCP-read workers. The job reads, checks and encodes the response, and the event loop
only wraps its bytes. A worker stays held until its job returns, even after its request is
cancelled or times out, so at most two checks run at once in the process. One more is refused at
submit and answers retryable 503 `service_unavailable` with `Retry-After: 5`
(`OntologyCheckBusy`). A read past the check's deadline or a failed read answers as the GET
routes answer it, with no findings. A check whose findings pass `CHECK_MAX_FINDINGS` stops and
is refused 400 `request_invalid` with pointer `/rows`.

`GET /api/graph/ontology/provenances` serves `OntologyProvenances`, built by
`graph_api.ontology_provenances.build_provenances` from the bound code index and the served
coverage index with no graph read. `GraphDeps.ontology_provenances`, a single-flight cache of
one entry keyed by the code index's content hash and the coverage build
(`CoverageManifest.build_id`), holds the frozen body, and the lifespan registers it as
`graph_api.ontology_provenances`, so a generation change clears it too. A miss builds the body
in a worker thread, and concurrent misses share that build. It answers as the other two routes
do.
"""

from __future__ import annotations

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
from dfo_platform.refusal import TIMEOUT_DEFAULT
from dfo_platform.single_flight_cache import SingleFlightCache, WaitTimeout
from dfo_platform.static_response import FrozenJSON, freeze_public_json
from graph_api.config import (
    GRAPH_NODE_TIMEOUT_SECONDS,
    GRAPH_ONTOLOGY_CHECK_CONCURRENCY,
    GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS,
    ONTOLOGY_FLIGHT_TIMEOUT_SECONDS,
)
from graph_api.deps import (
    DcpExecutorSaturated,
    GraphDeps,
    GraphDepsDep,
    log_queue_wait,
    run_dcp_read,
)
from graph_api.ontology_check import (
    CheckReads,
    OntologyCheckTooManyFindings,
    OntologyCheckTooManyReads,
    OntologyCheckUnservedId,
    OntologyLookup,
    check_rows,
    plan_reads,
)
from graph_api.ontology_document import (
    OntologyListingTruncatedError,
    build_ontology,
)
from graph_api.ontology_models import (
    CHECK_MAX_READ_IDS,
    OntologyCheckRequest,
    OntologyCheckResponse,
    OntologyCheckRow,
    OntologyDocument,
    OntologyProvenances,
    OntologyTerms,
)
from graph_api.ontology_provenances import ProvenanceCoverage, build_provenances
from graph_api.reader import GraphNodeReader
from graph_generation import GenerationProvider
from graph_generation.stamp import UNKNOWN_KEY
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from origin_catalogue import current_catalogue

ONTOLOGY_RETRY_AFTER_SECONDS = 5
"""The `Retry-After` a request sends when its deadline passes while the build runs on."""

_MAX_AGE_SECONDS = 300


class OntologyKey(NamedTuple):
    """The graph generation token, `UNKNOWN_KEY` where it is unknown, and the content hash of
    the code index whose StatVar rows the build read."""

    generation: str
    code_index: str


@dataclass(frozen=True, slots=True)
class FrozenOntology:
    """Both bodies of one build, each encoded once, the build's generation and the lookup the
    check reads."""

    document: FrozenJSON
    terms: FrozenJSON
    generation: str | None
    lookup: OntologyLookup


@dataclass(frozen=True, slots=True)
class RefusedOntology:
    """A build refusal that cannot change while the generation holds, a truncated term listing.
    `error` makes a fresh exception for each request it answers."""

    error: Callable[[], GraphReadError]


type OntologyEntry = FrozenOntology | RefusedOntology


class OntologyBuildPending(Exception):
    """A request's deadline passed while the build it waits on runs on."""

    def __init__(self) -> None:
        super().__init__("the ontology is still being built; retry")


class OntologyBuildExpired(Exception):
    """The build outlived its flight deadline, so it stored nothing."""

    def __init__(self) -> None:
        super().__init__("the ontology build passed its deadline")


class OntologyCheckBusy(Exception):
    """Every worker of `GraphDeps.check_executor` holds a check job."""

    def __init__(self) -> None:
        super().__init__(
            f"{GRAPH_ONTOLOGY_CHECK_CONCURRENCY} ontology checks are already running; retry"
        )


class OntologyReadFailed(Exception):
    """A graph read of the build or of the check failed with `error`."""

    def __init__(self, error: GraphReadError) -> None:
        super().__init__(f"an ontology graph read failed: {error}")
        self.error = error


class ProvenancesKey(NamedTuple):
    """The content hash of the code index and the coverage build the provenances body reads."""

    code_index: str
    coverage_build: str


def new_provenances_cache(
    *, generation: GenerationProvider
) -> SingleFlightCache[ProvenancesKey, FrozenJSON]:
    """The cache of the provenances body: one entry, whose build runs in a worker thread and
    whose key carries `generation`'s stamp."""
    return SingleFlightCache(
        maxsize=1, flight_timeout_s=ONTOLOGY_FLIGHT_TIMEOUT_SECONDS, generation=generation
    )


def new_ontology_cache(
    *, executor: Executor, generation: GenerationProvider
) -> SingleFlightCache[OntologyKey, OntologyEntry]:
    """The cache of the built bodies: one entry, whose build runs on `executor` and whose key
    carries `generation`'s stamp."""
    return SingleFlightCache(
        maxsize=1,
        flight_timeout_s=ONTOLOGY_FLIGHT_TIMEOUT_SECONDS,
        executor=executor,
        generation=generation,
    )


def _freeze(model: OntologyDocument | OntologyTerms | OntologyProvenances) -> FrozenJSON:
    return freeze_public_json(model.model_dump(mode="json"), max_age=_MAX_AGE_SECONDS)


def _request_deadline(seconds: float | None = None) -> float:
    """`seconds` from now, `GRAPH_NODE_TIMEOUT_SECONDS` when `None`."""
    return time.monotonic() + (GRAPH_NODE_TIMEOUT_SECONDS if seconds is None else seconds)


def _build_entry(
    deps: GraphDeps, *, code_index: CodeIndex, generation: str | None, deadline: float
) -> OntologyEntry:
    """Build and freeze the bodies, or keep a refusal that cannot change within the
    generation."""
    try:
        ontology = build_ontology(
            reader=deps.graph, code_index=code_index, generation=generation, deadline=deadline
        )
    except OntologyListingTruncatedError as exc:
        return RefusedOntology(error=partial(OntologyListingTruncatedError, str(exc)))
    return FrozenOntology(
        document=_freeze(ontology.document),
        terms=_freeze(ontology.terms),
        generation=ontology.document.generation,
        lookup=OntologyLookup.of(ontology.document),
    )


async def _frozen_ontology(
    deps: GraphDeps, *, code_index: CodeIndex, deadline: float
) -> FrozenOntology:
    """The built bodies for `code_index` from `deps.ontology`, building them on a miss and
    waiting until `deadline`. Refresh the generation first, as the documentation route does, so
    the stamp is fresh after an idle gap.

    Raises:
        OntologyReadFailed: the build failed, or the entry keeps its refusal.
    """
    await deps.generation.refresh_if_stale()
    token = deps.generation.stamp().token
    key = OntologyKey(
        generation=token if token is not None else UNKNOWN_KEY,
        code_index=code_index.stamp.content_hash,
    )

    def build(_keys: list[OntologyKey], flight_deadline: float) -> dict[OntologyKey, OntologyEntry]:
        entry = _build_entry(
            deps, code_index=code_index, generation=token, deadline=flight_deadline
        )
        return {key: entry}

    try:
        found = await deps.ontology.aget_or_fetch_many(
            [key], log_queue_wait(build), deadline=deadline
        )
    except WaitTimeout:
        raise OntologyBuildPending() from None
    except TimeoutError as exc:
        raise OntologyBuildExpired() from exc
    except GraphReadError as exc:
        raise OntologyReadFailed(exc) from exc
    entry = found[key]
    if isinstance(entry, RefusedOntology):
        raise OntologyReadFailed(entry.error())
    return entry


def classify_ontology_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify the build's waits and read failures, which this door raises as its own types,
    and the check's caps and concurrency limit."""
    if isinstance(
        exc, OntologyCheckTooManyReads | OntologyCheckUnservedId | OntologyCheckTooManyFindings
    ):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=str(exc),
            pointer="/rows",
            retryable=False,
        )
    if isinstance(exc, OntologyBuildPending | OntologyCheckBusy):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=str(exc),
            retryable=True,
            headers={"Retry-After": str(ONTOLOGY_RETRY_AFTER_SECONDS)},
        )
    if isinstance(exc, OntologyBuildExpired):
        return ProblemSpec(
            status=TIMEOUT_DEFAULT.status,
            code=TIMEOUT_DEFAULT.code,
            detail="the ontology build passed its deadline; retry",
            retryable=TIMEOUT_DEFAULT.retryable,
            log="Graph ontology door's build passed its flight deadline",
        )
    if not isinstance(exc, OntologyReadFailed):
        return None
    error = exc.error
    if isinstance(error, OntologyListingTruncatedError):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=str(error),
            retryable=False,
            log=f"Graph ontology door refused a truncated term listing: {error}",
        )
    if isinstance(error, GraphDeadlineExceededError):
        return ProblemSpec(
            status=TIMEOUT_DEFAULT.status,
            code=TIMEOUT_DEFAULT.code,
            detail="the deadline expired during the ontology door's graph reads; retry",
            retryable=TIMEOUT_DEFAULT.retryable,
        )
    upstream_status = error.upstream_status
    client_side = isinstance(upstream_status, int) and 400 <= upstream_status < 500
    spec = (
        ProblemSpec(status=502, code="upstream_error", retryable=True)
        if client_side
        else ProblemSpec(status=503, code="service_unavailable", retryable=True)
    )
    return spec._replace(log=f"Graph ontology door's DCP read failed: {error}", log_traceback=True)


def _freeze_provenances(code_index: CodeIndex, coverage: ProvenanceCoverage) -> FrozenJSON:
    return _freeze(build_provenances(code_index=code_index, coverage=coverage))


async def _frozen_provenances(deps: GraphDeps) -> FrozenJSON:
    """The provenances body of the bound code index and the served coverage build, from
    `deps.ontology_provenances`. A miss builds and encodes the body in a worker thread, and
    concurrent misses share that one build. A request waits until `GRAPH_NODE_TIMEOUT_SECONDS`
    from its start.

    Raises:
        OntologyBuildPending: the request's wait passed while the build runs on.
        OntologyBuildExpired: the build outlived its flight deadline.
    """
    code_index = get_index()
    snapshot = deps.coverage.snapshot()
    key = ProvenancesKey(
        code_index=code_index.stamp.content_hash, coverage_build=snapshot.manifest.build_id
    )
    try:
        return await deps.ontology_provenances.aget_or_fetch(
            key,
            lambda _flight_deadline: _freeze_provenances(code_index, snapshot.index),
            deadline=_request_deadline(),
        )
    except WaitTimeout:
        raise OntologyBuildPending() from None
    except TimeoutError as exc:
        raise OntologyBuildExpired() from exc


async def get_ontology(deps: GraphDepsDep, request: Request) -> Response:
    frozen = await _frozen_ontology(deps, code_index=get_index(), deadline=_request_deadline())
    return frozen.document.respond(request)


async def get_ontology_terms(deps: GraphDepsDep, request: Request) -> Response:
    frozen = await _frozen_ontology(deps, code_index=get_index(), deadline=_request_deadline())
    return frozen.terms.respond(request)


def _check_body(
    rows: Sequence[OntologyCheckRow],
    *,
    reads: CheckReads,
    lookup: OntologyLookup,
    reader: GraphNodeReader,
    code_index: CodeIndex,
    generation: str | None,
    deadline: float,
) -> bytes:
    """The encoded response of one check, from `check_rows` and `OntologyCheckResponse`'s JSON."""
    findings = check_rows(
        rows, reads=reads, lookup=lookup, reader=reader, code_index=code_index, deadline=deadline
    )
    response = OntologyCheckResponse(generation=generation, row_count=len(rows), findings=findings)
    return response.model_dump_json().encode()


async def post_ontology_check(deps: GraphDepsDep, body: OntologyCheckRequest) -> Response:
    deadline = _request_deadline(GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS)
    code_index = get_index()
    frozen = await _frozen_ontology(deps, code_index=code_index, deadline=deadline)
    reads = plan_reads(
        body.rows, lookup=frozen.lookup, un_served=current_catalogue().un_system_enabled
    )
    try:
        encoded = await run_dcp_read(
            deps.check_executor,
            _check_body,
            body.rows,
            reads=reads,
            lookup=frozen.lookup,
            reader=deps.graph,
            code_index=code_index,
            generation=frozen.generation,
            deadline=deadline,
        )
    except DcpExecutorSaturated:
        raise OntologyCheckBusy() from None
    except GraphReadError as exc:
        raise OntologyReadFailed(exc) from exc
    return Response(content=encoded, media_type="application/json")


async def get_ontology_provenances(deps: GraphDepsDep, request: Request) -> Response:
    return (await _frozen_provenances(deps)).respond(request)


def build_ontology_door() -> DoorContribution:
    router = APIRouter()
    # `response_model` documents the response body in OpenAPI. The handlers' responses pass
    # through.
    router.add_api_route(
        "/api/graph/ontology", get_ontology, methods=["GET"], response_model=OntologyDocument
    )
    router.add_api_route(
        "/api/graph/ontology/terms",
        get_ontology_terms,
        methods=["GET"],
        response_model=OntologyTerms,
    )
    router.add_api_route(
        "/api/graph/ontology/provenances",
        get_ontology_provenances,
        methods=["GET"],
        response_model=OntologyProvenances,
    )
    router.add_api_route(
        "/api/graph/ontology/check",
        post_ontology_check,
        methods=["POST"],
        response_model=OntologyCheckResponse,
    )
    return DoorContribution(
        router=router,
        classify=classify_ontology_exception,
        exception_types=(
            OntologyBuildPending,
            OntologyBuildExpired,
            OntologyReadFailed,
            OntologyCheckTooManyReads,
            OntologyCheckUnservedId,
            OntologyCheckTooManyFindings,
            OntologyCheckBusy,
        ),
        exception_exemplars=(
            OntologyCheckTooManyReads(CHECK_MAX_READ_IDS + 1),
            OntologyCheckUnservedId("undata/exemplar"),
            OntologyCheckTooManyFindings(),
            OntologyCheckBusy(),
            OntologyBuildPending(),
            OntologyBuildExpired(),
            OntologyReadFailed(OntologyListingTruncatedError("exemplar")),
            OntologyReadFailed(GraphDeadlineExceededError("exemplar")),
            OntologyReadFailed(GraphReadError("exemplar", upstream_status=400)),
            OntologyReadFailed(GraphReadError("exemplar", upstream_status=503)),
        ),
    )
