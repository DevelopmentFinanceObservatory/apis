"""Serve the instance's ontology document and the list of every Class and Property.

`GET /api/graph/ontology` serves `OntologyDocument` and `GET /api/graph/ontology/terms`
serves `OntologyTerms`. One build (`graph_api.ontology_document.build_ontology`) yields both,
and `GraphDeps.ontology` holds its result: a single-flight cache of one entry, keyed by the
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
cap or a typed node read with no declaring provenance non-retryable 503 `service_unavailable`, a DCP 4xx 502 `upstream_error` and any other read
failure retryable 503. A failed build is not kept.

The cached value holds each body encoded once with its keys sorted, so two builds of one graph
give the same bytes. Each answers `public, max-age=300` with a strong `ETag` over those bytes
and a gzip representation, honouring `If-None-Match` with 304
(`dfo_platform.static_response.FrozenJSON`).

`POST /api/graph/ontology/check` takes 1 to `CHECK_MAX_ROWS` candidate StatVar definitions and
answers every fault `graph_api.ontology_check.check_rows` finds in them, with the generation
of the build it checked against. The entry keeps, beside its frozen bodies, the build's
`OntologyLookup`, built once per entry, and the code index the build read, so the check
compares served definitions against the index the build read. The check's deadline is
`GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS` from its start. It waits on the same flight as the GET
routes until that deadline and answers their 503 and 504 the same way. The check's own reads,
of at most `CHECK_MAX_READ_IDS` ids the document does not hold, run as one job on the DCP-read
executor under the same deadline; a saturated executor, a read past the deadline or a failed
read answers as the GET routes answer it, with no findings. A body outside the request model
is refused 400 `request_invalid`: `rows` empty or longer than `CHECK_MAX_ROWS`, a row past
`CHECK_MAX_CONSTRAINT_PROPERTIES` or `CHECK_MAX_OBSERVATION_PROPERTIES`, a `ref` or a constraint
value string past 512 characters, or a constraint value that is no finite number. Rows naming
more than `CHECK_MAX_READ_IDS` ids to read are refused the same way before any read, with
pointer `/rows`.

`GET /api/graph/ontology/provenances` serves `OntologyProvenances`, built by
`graph_api.ontology_provenances.build_provenances` from the bound code index and the served
coverage index with no graph read. The frozen body is kept per code index (`per_index`) for the
coverage build it was read with (`CoverageManifest.build_id`): a request with another build
rebuilds it in a worker thread, and the entry goes with its code index. It answers as the other
two routes do.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import NamedTuple

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from code_index import CodeIndex
from code_index.shared import get_index, per_index
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.refusal import TIMEOUT_DEFAULT
from dfo_platform.single_flight_cache import SingleFlightCache, WaitTimeout
from dfo_platform.static_response import FrozenJSON, freeze_public_json
from graph_api.config import (
    GRAPH_NODE_TIMEOUT_SECONDS,
    GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS,
    ONTOLOGY_FLIGHT_TIMEOUT_SECONDS,
)
from graph_api.deps import GraphDeps, GraphDepsDep, log_queue_wait, run_dcp_read
from graph_api.ontology_check import OntologyCheckTooManyReads, OntologyLookup, check_rows
from graph_api.ontology_document import (
    OntologyListingTruncatedError,
    OntologyProvenanceUnreadError,
    build_ontology,
)
from graph_api.ontology_models import (
    CHECK_MAX_READ_IDS,
    OntologyCheckRequest,
    OntologyCheckResponse,
    OntologyDocument,
    OntologyProvenances,
    OntologyTerms,
)
from graph_api.ontology_provenances import ProvenanceCoverage, build_provenances
from graph_generation import GenerationProvider
from graph_generation.stamp import UNKNOWN_KEY
from materialise.graph import GraphDeadlineExceededError, GraphReadError

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
    check reads, and the code index the build read, which the check compares served definitions
    against. The entry holds the index only as long as the cache holds the entry."""

    document: FrozenJSON
    terms: FrozenJSON
    generation: str | None
    lookup: OntologyLookup
    code_index: CodeIndex


class OntologyBuildPending(Exception):
    """A request's deadline passed while the build it waits on runs on."""

    def __init__(self) -> None:
        super().__init__("the ontology is still being built; retry")


class OntologyBuildExpired(Exception):
    """The build outlived its flight deadline, so it stored nothing."""

    def __init__(self) -> None:
        super().__init__("the ontology build passed its deadline")


class OntologyReadFailed(Exception):
    """A graph read of the build or of the check failed with `error`."""

    def __init__(self, error: GraphReadError) -> None:
        super().__init__(f"an ontology graph read failed: {error}")
        self.error = error


def new_ontology_cache(
    *, executor: Executor, generation: GenerationProvider
) -> SingleFlightCache[OntologyKey, FrozenOntology]:
    """The cache of the built bodies: one entry, whose build runs on `executor` and whose key
    carries `generation`'s stamp."""
    return SingleFlightCache(
        maxsize=1,
        flight_timeout_s=ONTOLOGY_FLIGHT_TIMEOUT_SECONDS,
        executor=executor,
        generation=generation,
    )


def _by_key(entry: tuple[object, object]) -> str:
    return str(entry[0])


def _sorted_keys(value: object) -> object:
    """`value` with every mapping's keys in sorted order, recursively."""
    if isinstance(value, dict):
        return {key: _sorted_keys(item) for key, item in sorted(value.items(), key=_by_key)}
    if isinstance(value, list):
        return [_sorted_keys(item) for item in value]
    return value


def _freeze(model: OntologyDocument | OntologyTerms | OntologyProvenances) -> FrozenJSON:
    return freeze_public_json(_sorted_keys(model.model_dump(mode="json")), max_age=_MAX_AGE_SECONDS)


def _request_deadline(seconds: float | None = None) -> float:
    """`seconds` from now, `GRAPH_NODE_TIMEOUT_SECONDS` when `None`."""
    return time.monotonic() + (GRAPH_NODE_TIMEOUT_SECONDS if seconds is None else seconds)


async def _frozen_ontology(deps: GraphDeps, *, deadline: float) -> FrozenOntology:
    """The built bodies from `deps.ontology`, building them on a miss and waiting until
    `deadline`. Refresh the generation first, as the documentation route does, so the stamp is
    fresh after an idle gap."""
    await deps.generation.refresh_if_stale()
    code_index = get_index()
    token = deps.generation.stamp().token
    key = OntologyKey(
        generation=token if token is not None else UNKNOWN_KEY,
        code_index=code_index.stamp.content_hash,
    )

    def build(
        _keys: list[OntologyKey], flight_deadline: float
    ) -> dict[OntologyKey, FrozenOntology]:
        ontology = build_ontology(
            reader=deps.graph,
            code_index=code_index,
            generation=token,
            deadline=flight_deadline,
        )
        return {
            key: FrozenOntology(
                document=_freeze(ontology.document),
                terms=_freeze(ontology.terms),
                generation=ontology.document.generation,
                lookup=OntologyLookup.of(ontology.document),
                code_index=code_index,
            )
        }

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
    return found[key]


def classify_ontology_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify the build's waits and read failures, which this door raises as its own types,
    and the check's read cap."""
    if isinstance(exc, OntologyCheckTooManyReads):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=str(exc),
            pointer="/rows",
            retryable=False,
        )
    if isinstance(exc, OntologyBuildPending):
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
    if isinstance(error, OntologyProvenanceUnreadError):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=str(error),
            retryable=False,
            log=f"Graph ontology door refused {error.dcid!r}, read with no declaring provenance",
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


@dataclass(slots=True)
class _ProvenancesMemo:
    """One code index's frozen provenances body and the coverage build it was read with."""

    coverage_build: str | None = None
    body: FrozenJSON | None = None


_provenances_memo = per_index(lambda _index: _ProvenancesMemo())


def _freeze_provenances(code_index: CodeIndex, coverage: ProvenanceCoverage) -> FrozenJSON:
    return _freeze(build_provenances(code_index=code_index, coverage=coverage))


async def _frozen_provenances(deps: GraphDeps) -> FrozenJSON:
    """The provenances body of the bound code index and the served coverage build. A miss
    builds and encodes the body in a worker thread; two concurrent misses both build, and
    either stores the same body."""
    code_index = get_index()
    snapshot = deps.coverage.snapshot()
    memo = _provenances_memo(code_index)
    build_id = snapshot.manifest.build_id
    if memo.body is not None and memo.coverage_build == build_id:
        return memo.body
    body = await asyncio.to_thread(_freeze_provenances, code_index, snapshot.index)
    memo.body, memo.coverage_build = body, build_id
    return body


async def get_ontology(deps: GraphDepsDep, request: Request) -> Response:
    frozen = await _frozen_ontology(deps, deadline=_request_deadline())
    return frozen.document.respond(request)


async def get_ontology_terms(deps: GraphDepsDep, request: Request) -> Response:
    frozen = await _frozen_ontology(deps, deadline=_request_deadline())
    return frozen.terms.respond(request)


async def check_ontology(deps: GraphDepsDep, body: OntologyCheckRequest) -> JSONResponse:
    deadline = _request_deadline(GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS)
    frozen = await _frozen_ontology(deps, deadline=deadline)
    try:
        findings = await run_dcp_read(
            deps.dcp_executor,
            check_rows,
            body.rows,
            lookup=frozen.lookup,
            reader=deps.graph,
            code_index=frozen.code_index,
            deadline=deadline,
        )
    except GraphReadError as exc:
        raise OntologyReadFailed(exc) from exc
    response = OntologyCheckResponse(
        generation=frozen.generation, row_count=len(body.rows), findings=findings
    )
    return JSONResponse(response.model_dump(mode="json"))


async def get_ontology_provenances(deps: GraphDepsDep, request: Request) -> Response:
    return (await _frozen_provenances(deps)).respond(request)


def build_ontology_door() -> DoorContribution:
    router = APIRouter()
    # `response_model` documents the response body in OpenAPI; the handlers' responses pass
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
        check_ontology,
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
        ),
        exception_exemplars=(
            OntologyCheckTooManyReads(CHECK_MAX_READ_IDS + 1),
            OntologyBuildPending(),
            OntologyBuildExpired(),
            OntologyReadFailed(OntologyListingTruncatedError("exemplar")),
            OntologyReadFailed(OntologyProvenanceUnreadError("exemplar")),
            OntologyReadFailed(GraphDeadlineExceededError("exemplar")),
            OntologyReadFailed(GraphReadError("exemplar", upstream_status=400)),
            OntologyReadFailed(GraphReadError("exemplar", upstream_status=503)),
        ),
    )
