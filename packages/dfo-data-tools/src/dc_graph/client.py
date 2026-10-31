"""Synchronous Data Commons client for `/v2/node`, `/v2/observation` and `/v2/resolve`.

The client handles chunking, paging and false-empty retries. It implements the materialisation and
Graph API reader protocols. Search adds an adapter for Search-specific result types.

By default, reads get three attempts with 0.5 s and 2.0 s backoffs. 5xx responses and transport
errors retry. 4xx responses and non-JSON 2xx bodies fail. Search configures one attempt.
Sized caches are off by default, and entries are written only after successful reads. Each
response logs one line with its path, status and milliseconds to headers, at DEBUG, or at WARNING
past `_SLOW_RESPONSE_SECONDS` (`_log_response`).

Callers can supply read, deadline and cap error subclasses with matching constructor signatures.
The read and deadline classes must inherit their corresponding `GraphReadError` types. Cap errors
must inherit `GraphCapExceededError`. The constructor raises `TypeError` for any other class.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, replace
from functools import partial
from time import perf_counter
from types import MappingProxyType
from typing import Final, Literal, Self

import cachetools
import httpx

from dc_graph.errors import (
    CapErrorFactory,
    GraphCapExceededError,
    GraphDeadlineExceededError,
    GraphNotAdvancingError,
    GraphOmittedDcidsError,
    GraphReadError,
    ReadErrorFactory,
)
from dc_graph.visible_arcs import dfo_visible
from dfo_platform.cloud_run_auth import cloud_run_auth
from dfo_platform.executor import submit_with_context
from dfo_platform.request_deadline import bounded_deadline
from dfo_platform.single_flight_cache import SingleFlightCache, WaitTimeout
from dfo_platform.timings import count
from graph_generation import UNKNOWN_GENERATION, GenerationProvider, GenerationStamp
from graph_generation.stamp import ttu_from_stamp_key

logger = logging.getLogger(__name__)

_TOKEN_HEADER = "x-graph-token"
# The request extension `_count_request` stamps with the `perf_counter` time it sent the request,
# a clock kept apart from the `time.monotonic` deadlines.
_SENT_AT = "dfo_sent_at"
NODE_PATH: Final[str] = "/v2/node"
_OBSERVATION_PATH = "/v2/observation"
_RESOLVE_PATH = "/v2/resolve"
_DEADLINE_EXCEEDED_DETAIL = "the graph read exceeded its absolute caller deadline"
_ERROR_BODY_SNIPPET_BYTES = 4096
_FRONT_DOOR_REFUSAL_PAGE_MARKER = "<title>403</title>403 Forbidden"
_OBS_PROPERTIES_FOLLOWER_TIMEOUT_DETAIL = (
    "node_observation_properties_batch: declared-role refresh deadline elapsed waiting "
    "on the shared read"
)

_UNSAFE_FILTER_CHARS = frozenset(",:{}<>")
# Bracket-list arc specs reserve `[` and `]` as delimiters.
_UNSAFE_PROPERTY_CHARS = _UNSAFE_FILTER_CHARS | frozenset("[]")

UNRESOLVED_NODE: Final[str] = "\x00unresolved-node"
"""Marker for an arc entry with neither `dcid` nor `value`. The leading NUL distinguishes it from
real DCIDs and literal arc values.
"""

_NAME_PROPERTIES: Final[tuple[str, str]] = ("name", "alternateName")

CURRENCY_UNIT_OF_MEASURE_ISO_CODES: Final[Mapping[str, str]] = MappingProxyType(
    {"USDollar": "USD", "Euro": "EUR", "CanadianDollar": "CAD", "IndianRupee": "INR"}
)
"""ISO 4217 codes for the four plain currencies base Data Commons types `CurrencyUnitOfMeasure`.

The graph gives these nodes no `iso4217AlphaCode` and no `CurrencyUnit` type, so the
`currency_units` read never sees them, yet base Data Commons facets report units in them. Their
`shortDisplayName` is display text ("USD/hr" on a sibling), so it cannot stand in for the code.
`currency_units` merges this map under the graph's own codes; an ontology overlay that gives
these nodes an `iso4217AlphaCode` can replace it.
"""

_CURRENCY_CLASS: Final[str] = "Currency"
_ISO_4217_SHAPE: Final = re.compile(r"[A-Z]{3}")
"""A `Currency` member's dcid is its ISO 4217 code. A few legacy members carry four letters
(`MYAD`), which no ISO code has."""

# Paging limits raise on exhaustion, preserving complete reads.
_NODE_ARCS_MANY_MAX_PAGES = 200
_NODE_SUBJECTS_MAX_PAGES = 200
_NODE_SUBJECTS_STUCK_PAGE_LIMIT: Final[int] = 3
_NODE_SUBJECTS_MANY_MAX_PAGES = 200
_ENUM_MEMBERS_TYPEOF_MAX_PAGES = 200

SubjectsCoverageFlag = Literal["complete", "capped", "budget_exhausted"]
"""Per-lead end state from `node_subjects_many`: complete, capped, or stopped by the page budget.
"""

# Fixed backoffs make waits deterministic. Index 0 is the first retry.
_RETRY_BACKOFF_SECONDS: Final[tuple[float, ...]] = (0.5, 2.0)
# A response slower than this logs at WARNING instead of DEBUG (`_log_response`).
_SLOW_RESPONSE_SECONDS: Final[float] = 2.0

_DEFAULT_NODE_ARC_CHUNK_SIZE = 50
_DEFAULT_NODE_ARC_CALL_CONCURRENCY = 16
_DEFAULT_NODE_ARC_TOTAL_CONCURRENCY = 32
_DEFAULT_NODE_ARC_MANY_TOTAL_CONCURRENCY = 32
_DEFAULT_SINGLE_ARC_BATCH_CHUNK_SIZE = 500
_DEFAULT_NODE_ARC_BATCH_CHUNK_SIZE = 12
_DEFAULT_ENTITIES_WITH_DATA_VARIABLE_CHUNK_SIZE = 200

_ttu_from_stamp_key = ttu_from_stamp_key

_DEFAULT_OBSERVATION_PROPERTIES_FLIGHT_TIMEOUT_S: Final[float] = 60.0
"""`GraphClient`'s default deadline for a shared `node_observation_properties_batch` read, from
when it starts. Search, the only reader of observation properties, passes its own value. The
default bounds the read for every other constructor (graph_api, datasets_api, presentation_api,
the jobs)."""

_OBSERVATION_PROPERTIES_FLIGHT_WORKERS: Final[int] = 32
"""Threads that run shared observation-property reads, process-wide, one per concurrent read.
Each holds its thread mostly waiting on chunk POSTs, which the
node-arc semaphore bounds."""

_OBSERVATION_PROPERTIES_FLIGHTS: Final = ThreadPoolExecutor(
    max_workers=_OBSERVATION_PROPERTIES_FLIGHT_WORKERS, thread_name_prefix="obs-properties"
)
"""Runs every client's shared observation-property reads. No caller of them runs on it."""

_declared_role_refresh_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "dc_graph_declared_role_refresh_deadline", default=None
)


def is_front_door_refusal(*, status: int, content_type: str | None, body: str) -> bool:
    """Whether a response is the UN front door's refusal: HTTP 403 with the Cloud Armor default
    HTML page, whose body carries `<title>403</title>403 Forbidden`. It refuses observation GETs
    naming certain places, and some `/v2/node` POSTs whose dcids it accepts one by one. A 403 in
    any other shape comes from somewhere else and is an upstream error."""
    return (
        status == 403
        and (content_type or "").startswith("text/html")
        and _FRONT_DOOR_REFUSAL_PAGE_MARKER in body
    )


class FrontDoorRefused(Exception):
    """A request met the front door's refusal (`is_front_door_refusal`). `post` attaches it as the
    `__cause__` of the `GraphReadError` it raises, so an adapter reading through `post` recognises
    the refusal with `isinstance(exc.__cause__, FrontDoorRefused)`. Observation GETs catch it
    inside the client."""


@contextlib.contextmanager
def declared_role_refresh_deadline(deadline: float | None):
    """Bound how long `node_observation_properties_batch` waits, by an absolute monotonic
    deadline. Expiry raises `TimeoutError` in this caller only. The shared read runs on under its
    own deadline and publishes. `None` means no deadline beyond the bound request deadline.
    """
    token = _declared_role_refresh_deadline.set(deadline)
    try:
        yield
    finally:
        _declared_role_refresh_deadline.reset(token)


@dataclass(frozen=True)
class Facet:
    """One facet for a `(stat_var, entity)` pair. Reads by this client report `obs_count` as the
    mixer's `obsCount`, or 0 when it is absent.

    `dates` and `values` are index-aligned and hold one entry per dated observation. Both are
    empty unless the read selected dates. `dates_complete` is `False` only when a dated read
    returned fewer dated observations than `obs_count`.

    `reads_annual` is whether the facet's dates are evidence that it is annual. A facet as
    read judges its own dates (`_reads_annual`). Merging facets of one id
    (`_merge_facets_by_id`, `materialise.facets.merge_facets`) stores the AND of the merged
    facets' evidence in `annual_evidence`, never a judgement of the widened bounds, so one
    place dated by month keeps the merged facet from reading as annual. `annual_evidence` is
    `None` on a facet no merge built.

    `first_series_end` is the latest date of the series that ends first among those the facet
    holds: a merge across places or StatVars (`materialise.facets.merge_facets`) keeps the
    earliest of theirs, and a coverage facet its earliest row's. `None` when every series
    the facet holds ends at `latest_date` (`earliest_series_end`), as on a facet of one.
    """

    obs_count: int
    earliest_date: str | None
    latest_date: str | None
    facet_id: str | None = None
    provenance_id: str | None = None
    measurement_method: str | None = None
    unit: str | None = None
    scaling_factor: str | None = None
    observation_period: str | None = None
    dates: tuple[str, ...] = ()
    values: tuple[float | None, ...] = ()
    dates_complete: bool = True
    # True when the dates are coverage's years rather than the graph's own: they are bare
    # years whatever the facet's cadence (`materialise.facets._facet_from_coverage_candidate`,
    # `search.engine.retrieval.role_scope.facet_from_coverage`).
    dates_coarsened: bool = False
    annual_evidence: bool | None = None
    first_series_end: str | None = None

    @property
    def earliest_series_end(self) -> str | None:
        """The latest date of the series that ends first (`first_series_end`), or
        `latest_date` on a facet of one series."""
        return self.first_series_end if self.first_series_end is not None else self.latest_date

    @property
    def reads_annual(self) -> bool:
        """The merged evidence, or this facet's own dates' (`_reads_annual`)."""
        return self.annual_evidence if self.annual_evidence is not None else _reads_annual(self)


def _reads_annual(facet: Facet) -> bool:
    """Whether `facet`'s own dates are evidence that it is annual. With a complete date list
    (`dates_complete`), every listed date and each known bound must be a bare year, so a
    month inside bare-year bounds is caught. Without one, both bounds, as the graph reports
    them, must be bare years. A coverage facet's years are coverage's whatever its cadence
    (`dates_coarsened`), so they are no evidence."""
    if facet.dates_coarsened:
        return False
    bounds = tuple(date for date in (facet.earliest_date, facet.latest_date) if date is not None)
    if facet.dates and facet.dates_complete:
        dates = (*bounds, *facet.dates)
    elif len(bounds) == 2:
        dates = bounds
    else:
        return False
    return all(len(date) == 4 and date.isdigit() for date in dates)


@dataclass(frozen=True)
class FacetDefaultPolicy:
    """Declared default-facet policy and resolved value for one axis."""

    resolved_ref: str
    policy_node: str
    policy_version: str
    selection_reason: str


def _count_request(request: httpx.Request) -> None:
    """Count each request this client sends, retries included, on the calling door's timings
    as `graph_requests`, and note when it was sent for `_log_response`."""
    count("graph_requests")
    request.extensions[_SENT_AT] = perf_counter()


def _log_response(response: httpx.Response) -> None:
    """Log each response with its method, path, status and the milliseconds from sending the
    request to its response headers, retries each on their own line. The line is DEBUG, or
    WARNING with the threshold when the response took longer than `_SLOW_RESPONSE_SECONDS`."""
    request = response.request
    elapsed_ms = (perf_counter() - request.extensions[_SENT_AT]) * 1000
    threshold_ms = _SLOW_RESPONSE_SECONDS * 1000
    if elapsed_ms > threshold_ms:
        logger.warning(
            "graph %s %s answered %d in %.0f ms, past the %.0f ms slow-response threshold",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            threshold_ms,
        )
    else:
        logger.debug(
            "graph %s %s answered %d in %.0f ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )


@dataclass(frozen=True)
class ProvenanceRefresh:
    """A provenance's `lastDataRefreshDate`, `None` if the node or the arc is absent."""

    last_data_refresh_date: str | None


@dataclass(frozen=True)
class SubjectPage:
    """One `node_subjects` page. `truncated` means `node_subjects` found a `(max_results + 1)`th
    distinct subject. Stuck cursors raise `GraphNotAdvancingError`.
    """

    dcids: tuple[str, ...]
    truncated: bool


@dataclass
class _CacheCounters:
    """Process-lifetime hit and miss counters for one cache. `clear()` leaves them unchanged."""

    hits: int = 0
    misses: int = 0


class _ChunkDeadlineExpired(Exception):
    """Internal signal when a chunk's semaphore wait reaches or outlasts its deadline.
    `node_observation_properties_batch` converts it to `TimeoutError`, and `node_arcs_batch`
    to the configured deadline error.
    """


class GraphClient:
    """Synchronous reader for the mixer's three `/v2` endpoints. One `httpx.Client` serves all
    methods and concurrent callers. Async callers should use `asyncio.to_thread`. `base_url` is the
    API root, with endpoint paths appended directly.

    Two process-wide pools are independent. `node_arc_total_concurrency` bounds
    `node_arcs_batch` and `node_observation_properties_batch`. `node_arc_many_total_concurrency`
    bounds `node_arcs_many` and `node_names_many` across all concurrent calls. Within one
    `node_arcs_many` call, `node_arc_chunk_size` limits dcids per POST and
    `node_arc_call_concurrency` limits in-flight chunks. `single_arc_batch_chunk_size` limits label
    and type batches, which use no pool and run `node_arc_call_concurrency` chunks at a time.
    `node_arc_batch_chunk_size` limits arc and observation-property batches, which halve chunks on
    continuation.

    Sized caches default to 0 (off). `single_value_cache_enabled` gates the Earth-country and
    currency-unit memos. A `GenerationProvider` keys entries by stamp and sets their maximum age.

    `/v2/observation` reads go by POST unless `observation_get_url_budget_bytes` is set. Then they
    go by GET, one request per group of variables and entities whose encoded query string fits
    the budget, run in sequence and merged. The UN System Data Commons answers `/v2/observation`
    by GET only, and its front door refuses a query string naming certain places with HTTP 403
    and a fixed HTML page (`is_front_door_refusal`). A GET that meets that refusal halves its
    entities until each refused entity stands alone, serves the rest, and leaves the refused
    ones out of later reads under the same generation (`blocked_observation_entities`). The same
    front door refuses some `/v2/node` POSTs naming several dcids that it serves one by one, so a
    batched node read that meets the refusal halves its dcids (`_read_refused_halves`). Any other
    failure raises, a 403 in another shape included. Node and resolve reads always POST.

    Facet reads take `domains`, sent as `filter.domains`: the mixer then returns only facets whose
    provenance URL hostname ends with one of them. Presence reads select no facet, and the mixer
    ignores the filter there.

    A request to a `*.run.app` host carries a Cloud Run ID token (`dfo_platform.cloud_run_auth`).
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        user_agent: str | None = None,
        max_attempts: int = 3,
        generation: GenerationProvider = UNKNOWN_GENERATION,
        timer: Callable[[], float] | None = None,
        node_arc_chunk_size: int = _DEFAULT_NODE_ARC_CHUNK_SIZE,
        node_arc_call_concurrency: int = _DEFAULT_NODE_ARC_CALL_CONCURRENCY,
        node_arc_total_concurrency: int = _DEFAULT_NODE_ARC_TOTAL_CONCURRENCY,
        node_arc_many_total_concurrency: int = _DEFAULT_NODE_ARC_MANY_TOTAL_CONCURRENCY,
        single_arc_batch_chunk_size: int = _DEFAULT_SINGLE_ARC_BATCH_CHUNK_SIZE,
        node_arc_batch_chunk_size: int = _DEFAULT_NODE_ARC_BATCH_CHUNK_SIZE,
        entities_with_data_variable_chunk_size: int = (
            _DEFAULT_ENTITIES_WITH_DATA_VARIABLE_CHUNK_SIZE
        ),
        label_cache_size: int = 0,
        type_cache_size: int = 0,
        arcs_cache_size: int = 0,
        obs_properties_cache_size: int = 0,
        observation_properties_flight_timeout_s: float = (
            _DEFAULT_OBSERVATION_PROPERTIES_FLIGHT_TIMEOUT_S
        ),
        obs_cache_size: int = 0,
        single_value_cache_enabled: bool = False,
        observation_get_url_budget_bytes: int | None = None,
        read_error_cls: ReadErrorFactory = GraphReadError,
        deadline_error_cls: ReadErrorFactory = GraphDeadlineExceededError,
        cap_error_cls: CapErrorFactory = GraphCapExceededError,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        if node_arc_chunk_size < 1:
            raise ValueError(f"node_arc_chunk_size must be at least 1, got {node_arc_chunk_size}")
        if node_arc_call_concurrency < 1:
            raise ValueError(
                f"node_arc_call_concurrency must be at least 1, got {node_arc_call_concurrency}"
            )
        if node_arc_total_concurrency < node_arc_call_concurrency:
            raise ValueError(
                f"node_arc_total_concurrency ({node_arc_total_concurrency!r}) must be >= "
                f"node_arc_call_concurrency ({node_arc_call_concurrency!r})"
            )
        if node_arc_many_total_concurrency < 1:
            raise ValueError(
                "node_arc_many_total_concurrency must be at least 1, got "
                f"{node_arc_many_total_concurrency}"
            )
        if not (isinstance(read_error_cls, type) and issubclass(read_error_cls, GraphReadError)):
            raise TypeError(
                f"read_error_cls must be a dc_graph.GraphReadError subclass, got {read_error_cls!r}"
            )
        if not (
            isinstance(deadline_error_cls, type)
            and issubclass(deadline_error_cls, GraphDeadlineExceededError)
        ):
            raise TypeError(
                "deadline_error_cls must be a dc_graph.GraphDeadlineExceededError subclass, "
                f"got {deadline_error_cls!r}"
            )
        if not (
            isinstance(cap_error_cls, type) and issubclass(cap_error_cls, GraphCapExceededError)
        ):
            raise TypeError(
                "cap_error_cls must be a dc_graph.GraphCapExceededError subclass, got "
                f"{cap_error_cls!r}"
            )
        _base = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if user_agent:
            headers["User-Agent"] = user_agent
        if token:
            headers[_TOKEN_HEADER] = token
        # Size the HTTP pool for both independent semaphores to saturate at once.
        _total_node_arc_slots = node_arc_total_concurrency + node_arc_many_total_concurrency
        self._client = httpx.Client(
            base_url=_base,
            headers=headers,
            auth=cloud_run_auth(),
            timeout=timeout,
            transport=transport,
            limits=httpx.Limits(
                max_connections=_total_node_arc_slots * 6,
                max_keepalive_connections=_total_node_arc_slots * 2,
            ),
            event_hooks={"request": [_count_request], "response": [_log_response]},
        )
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._generation = generation
        self._timer: Callable[[], float] = timer if timer is not None else time.monotonic
        self._node_arc_chunk_size = node_arc_chunk_size
        self._node_arc_call_concurrency = node_arc_call_concurrency
        self._single_arc_batch_chunk_size = single_arc_batch_chunk_size
        self._node_arc_batch_chunk_size = node_arc_batch_chunk_size
        self._entities_with_data_variable_chunk_size = entities_with_data_variable_chunk_size
        self._observation_get_url_budget_bytes = observation_get_url_budget_bytes
        self._node_arc_semaphore = threading.BoundedSemaphore(node_arc_total_concurrency)
        self._node_arc_many_semaphore = threading.BoundedSemaphore(node_arc_many_total_concurrency)
        self._read_error_cls = read_error_cls
        self._deadline_error_cls = deadline_error_cls
        self._cap_error_cls = cap_error_cls

        self._cache_lock = threading.Lock()
        self._label_cache: cachetools.TLRUCache[tuple[str, str], str | None] | None = (
            cachetools.TLRUCache(
                maxsize=label_cache_size, ttu=_ttu_from_stamp_key, timer=self._timer
            )
            if label_cache_size > 0
            else None
        )
        self._label_counters = _CacheCounters()
        self._type_cache: cachetools.TLRUCache[tuple[str, str], str | None] | None = (
            cachetools.TLRUCache(
                maxsize=type_cache_size, ttu=_ttu_from_stamp_key, timer=self._timer
            )
            if type_cache_size > 0
            else None
        )
        self._type_counters = _CacheCounters()
        self._arcs_cache: cachetools.TLRUCache[tuple[str, str], dict | None] | None = (
            cachetools.TLRUCache(
                maxsize=arcs_cache_size, ttu=_ttu_from_stamp_key, timer=self._timer
            )
            if arcs_cache_size > 0
            else None
        )
        self._arcs_counters = _CacheCounters()
        self._obs_cache: (
            cachetools.TLRUCache[tuple[str, str, str, bool, tuple[str, ...] | None], list[Facet]]
            | None
        ) = (
            cachetools.TLRUCache(maxsize=obs_cache_size, ttu=_ttu_from_stamp_key, timer=self._timer)
            if obs_cache_size > 0
            else None
        )
        self._obs_counters = _CacheCounters()
        # Keep single-flight coalescing active at cache size 0 with a one-entry store.
        self._observation_properties_flight: SingleFlightCache[str, tuple[str, ...]] = (
            SingleFlightCache(
                maxsize=max(obs_properties_cache_size, 1),
                flight_timeout_s=observation_properties_flight_timeout_s,
                executor=_OBSERVATION_PROPERTIES_FLIGHTS,
                generation=generation,
                clock=self._timer,
                timeout_message=_OBS_PROPERTIES_FOLLOWER_TIMEOUT_DETAIL,
            )
        )

        self._single_value_cache_enabled = single_value_cache_enabled
        self._earth_countries_cache: tuple[GenerationStamp, float, list[str]] | None = None
        self._earth_countries_counters = _CacheCounters()
        self._currency_units_cache: tuple[GenerationStamp, float, dict[str, str]] | None = None
        self._currency_units_counters = _CacheCounters()
        self._blocked_observation_entities: tuple[str, frozenset[str]] = ("", frozenset())

    def close(self) -> None:
        """Close the underlying httpx client. Safe to call more than once."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # Adapters use these methods for custom reads through the shared transport and retry logic.

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict:
        """POST with retries for 5xx responses and transport errors, using `_RETRY_BACKOFF_SECONDS`
        up to `max_attempts`. 4xx responses and non-JSON 2xx bodies fail immediately.

        An absolute monotonic `deadline` bounds each request timeout and retry wait. An exhausted
        budget fails before the request. A timeout raises the deadline error when the deadline bound
        the attempt or expired before the exception arrived. Otherwise, it remains retryable.
        """
        return self._request("POST", path, payload=payload, deadline=deadline)

    def _post_node(self, payload: dict, *, deadline: float | None) -> dict | None:
        """POST a `/v2/node` read through `post`. Returns `None` when the front door refuses it
        (`is_front_door_refusal`), which `_request` records as the read error's cause."""
        try:
            return self.post(NODE_PATH, payload, deadline=deadline)
        except GraphReadError as exc:
            if isinstance(exc.__cause__, FrontDoorRefused):
                return None
            raise

    def _read_refused_halves(
        self, dcids: Sequence[str], read: Callable[[Sequence[str]], dict]
    ) -> dict:
        """Read `dcids` as two halves with `read` after the front door refused a `/v2/node` POST
        naming them all, and merge the per-dcid results. A refusal naming one dcid raises the
        read error."""
        if len(dcids) == 1:
            raise self._read_error_cls(
                f"graph front door refused a /v2/node read naming {dcids[0]!r}",
                upstream_status=403,
            )
        mid = len(dcids) // 2
        return {**read(dcids[:mid]), **read(dcids[mid:])}

    def post_capped_bytes(
        self, path: str, payload: dict, *, max_bytes: int | None, deadline: float | None = None
    ) -> dict:
        """Stream the decoded body and raise the cap error as soon as it exceeds `max_bytes`.
        `None` delegates to `post`. The streamed path makes one attempt, so oversized bodies and
        5xx responses fail immediately.
        """
        if max_bytes is None:
            return self.post(path, payload, deadline=deadline)
        body, _ = self._request_capped(
            "POST", path, payload=payload, max_bytes=max_bytes, deadline=deadline
        )
        return body

    def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        payload: dict | None = None,
        params: httpx.QueryParams | None = None,
        deadline: float | None = None,
        front_door_refusals: bool = False,
    ) -> dict:
        """Send one request with `post`'s retry and deadline rules. With `front_door_refusals`,
        the front door's refusal raises `FrontDoorRefused`; without it, the read error it raises
        has a `FrontDoorRefused` cause (`_post_node`). A bound request deadline narrows
        `deadline` (`dfo_platform.request_deadline`)."""
        deadline = bounded_deadline(deadline)
        attempt = 0
        while True:
            attempt += 1
            request_timeout: float | None = None
            deadline_bound_attempt = False
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._deadline_error_cls(_DEADLINE_EXCEEDED_DETAIL)
                request_timeout = min(self._timeout, remaining)
                deadline_bound_attempt = remaining <= self._timeout
            try:
                send = (
                    partial(self._client.post, path, json=payload)
                    if method == "POST"
                    else partial(self._client.get, path, params=params)
                )
                response = send() if request_timeout is None else send(timeout=request_timeout)
            except httpx.TimeoutException as exc:
                if deadline is not None and (
                    deadline_bound_attempt or time.monotonic() >= deadline
                ):
                    raise self._deadline_error_cls(
                        f"{_DEADLINE_EXCEEDED_DETAIL} calling {path!r}: {exc}"
                    ) from exc
                if self._await_retry(attempt, deadline):
                    continue
                raise self._read_error_cls(
                    f"graph transport error calling {path!r}: {exc}"
                ) from exc
            except httpx.RequestError as exc:
                if self._await_retry(attempt, deadline):
                    continue
                raise self._read_error_cls(
                    f"graph transport error calling {path!r}: {exc}"
                ) from exc
            if not response.is_success:
                if 500 <= response.status_code < 600 and self._await_retry(attempt, deadline):
                    continue
                refused = is_front_door_refusal(
                    status=response.status_code,
                    content_type=response.headers.get("content-type"),
                    body=response.text,
                )
                if refused and front_door_refusals:
                    raise FrontDoorRefused(path)
                error = self._read_error_cls(
                    f"graph returned HTTP {response.status_code} for {path!r}",
                    upstream_status=response.status_code,
                )
                if refused:
                    raise error from FrontDoorRefused(path)
                raise error
            try:
                return response.json()
            except ValueError as exc:
                raise self._read_error_cls(
                    f"graph returned a non-JSON body for {path!r}: {exc}",
                    upstream_status=response.status_code,
                ) from exc

    def _request_capped(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        payload: dict | None = None,
        params: httpx.QueryParams | None = None,
        max_bytes: int,
        already_read: int = 0,
        deadline: float | None = None,
        front_door_refusals: bool = False,
    ) -> tuple[dict, int]:
        """Send one streamed request and return its body with its byte count. The cap error fires
        once `already_read` plus this body's bytes exceed `max_bytes`, so split reads share one cap.
        `front_door_refusals` and the bound request deadline follow `_request`.
        """
        deadline = bounded_deadline(deadline)
        request_timeout = self._timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._deadline_error_cls(_DEADLINE_EXCEEDED_DETAIL)
            request_timeout = min(self._timeout, remaining)
        try:
            open_stream = (
                partial(self._client.stream, "POST", path, json=payload)
                if method == "POST"
                else partial(self._client.stream, "GET", path, params=params)
            )
            with open_stream(timeout=request_timeout) as resp:
                if resp.status_code != 200:
                    snippet = bytearray()
                    for chunk in resp.iter_bytes():
                        snippet.extend(chunk)
                        if len(snippet) >= _ERROR_BODY_SNIPPET_BYTES:
                            break
                    if front_door_refusals and is_front_door_refusal(
                        status=resp.status_code,
                        content_type=resp.headers.get("content-type"),
                        body=bytes(snippet).decode("utf-8", errors="replace"),
                    ):
                        raise FrontDoorRefused(path)
                    raise self._read_error_cls(
                        f"graph returned HTTP {resp.status_code} for {path!r}: "
                        f"{bytes(snippet[:_ERROR_BODY_SNIPPET_BYTES])!r}",
                        upstream_status=resp.status_code,
                    )
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if already_read + total > max_bytes:
                        raise self._cap_error_cls(
                            f"graph read from {path!r} exceeded the byte cap of "
                            f"{max_bytes} (observed {already_read + total} bytes before "
                            "aborting)",
                            cap_kind="bytes",
                            cap=max_bytes,
                            observed=already_read + total,
                        )
                    chunks.append(chunk)
        except httpx.TimeoutException as exc:
            raise self._deadline_error_cls(
                f"{_DEADLINE_EXCEEDED_DETAIL} calling {path!r}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise self._read_error_cls(f"graph transport error calling {path!r}: {exc}") from exc
        try:
            return json.loads(b"".join(chunks)), total
        except ValueError as exc:
            raise self._read_error_cls(
                f"graph returned a non-JSON body for {path!r}: {exc}"
            ) from exc

    def _await_retry(self, attempt: int, deadline: float | None) -> bool:
        if attempt >= self._max_attempts:
            return False
        backoff = _RETRY_BACKOFF_SECONDS[min(attempt - 1, len(_RETRY_BACKOFF_SECONDS) - 1)]
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= backoff:
                return False
        time.sleep(backoff)
        return True

    def node_label(self, dcid: str) -> str | None:
        """Return the node's `label_or_self_name`, or `None` for an absent or unnamed node. It
        shares the optional cache with `node_labels_batch`. A node the graph answered without a
        name caches as `None`; an unanswered dcid caches nothing.
        """
        stamp = self._generation.stamp()
        return self._cached_single_arc(
            dcid,
            arc_property="->name",
            field="value",
            cache=self._label_cache,
            counters=self._label_counters,
            pick=label_or_self_name,
            stamp=stamp,
        )

    def node_type(self, dcid: str) -> str | None:
        """Return the first `->typeOf` dcid, or `None`. Shares the optional type cache with
        `node_types_batch`.
        """
        stamp = self._generation.stamp()
        return self._cached_single_arc(
            dcid,
            arc_property="->typeOf",
            field="dcid",
            cache=self._type_cache,
            counters=self._type_counters,
            pick=_pick_first,
            stamp=stamp,
        )

    def _cached_single_arc(
        self,
        dcid: str,
        *,
        arc_property: str,
        field: str,
        cache: cachetools.TLRUCache | None,
        counters: _CacheCounters,
        pick: Callable[[list[str], str], str | None],
        stamp: GenerationStamp,
    ) -> str | None:
        cache_key = (stamp.key, dcid)
        if cache is not None:
            with self._cache_lock:
                if cache_key in cache:
                    counters.hits += 1
                    return cache[cache_key]
                counters.misses += 1
        body = self.post(NODE_PATH, {"nodes": [dcid], "property": arc_property})
        data = body.get("data", {})
        node_data = data.get(dcid)
        arcs = (node_data or {}).get("arcs", {}) if node_data is not None else {}
        result = None
        if arcs:
            arc_name = arc_property.removeprefix("->")
            values = [n.get(field) for n in arcs.get(arc_name, {}).get("nodes", [])]
            result = pick(values, dcid)
        if dcid in data and cache is not None:
            with self._cache_lock:
                cache[cache_key] = result
        return result

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        """Fetch each node's `label_or_self_name` in chunks of at most
        `single_arc_batch_chunk_size`. Omit dcids with none. Continuations trigger chunk halving,
        and results share `node_label`'s cache.
        """
        return self._single_arc_batch(
            dcids,
            arc_property="->name",
            arc_name="name",
            field="value",
            cache=self._label_cache,
            counters=self._label_counters,
            pick=label_or_self_name,
        )

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        """Fetch the first `->typeOf` dcid in chunks of at most `single_arc_batch_chunk_size`.
        Omit nodes without a type arc. Results share `node_type`'s cache.
        """
        return self._single_arc_batch(
            dcids,
            arc_property="->typeOf",
            arc_name="typeOf",
            field="dcid",
            cache=self._type_cache,
            counters=self._type_counters,
            pick=_pick_first,
        )

    def _single_arc_batch(
        self,
        dcids: list[str],
        *,
        arc_property: str,
        arc_name: str,
        field: str,
        cache: cachetools.TLRUCache | None,
        counters: _CacheCounters,
        pick: Callable[[list[str], str], str | None],
    ) -> dict[str, str]:
        """Fetch uncached dcids in chunks and merge hits with results in first-occurrence input
        order. A dcid the graph answered without a value is omitted from the result, and caches as
        `None` only when it was the one dcid read; a dcid missing from the response caches
        nothing. `cache=None` treats every dcid as a miss.
        """
        if not dcids:
            return {}
        stamp = self._generation.stamp()
        unique_dcids = list(dict.fromkeys(dcids))
        hits: dict[str, str | None] = {}
        if cache is not None:
            with self._cache_lock:
                misses = []
                for dcid in unique_dcids:
                    cache_key = (stamp.key, dcid)
                    if cache_key in cache:
                        counters.hits += 1
                        hits[dcid] = cache[cache_key]
                    else:
                        counters.misses += 1
                        misses.append(dcid)
        else:
            misses = unique_dcids
            counters.misses += len(misses)

        data = (
            self._fetch_arc_chunks(
                misses,
                property_arc=arc_property,
                chunk_size=self._single_arc_batch_chunk_size,
                semaphore=None,
            )
            if misses
            else {}
        )
        fetched: dict[str, str | None] = {}
        read_alone = len(misses) == 1
        for dcid in misses:
            node_data = data.get(dcid)
            if node_data is None:
                continue
            values = [
                n.get(field)
                for n in node_data.get("arcs", {}).get(arc_name, {}).get("nodes", [])
                if n.get(field)
            ]
            picked = pick(values, dcid)
            fetched[dcid] = picked
            # An empty entry from a multi-dcid chunk may depend on the chunk's composition, so
            # only a dcid read alone is trusted empty.
            if cache is not None and (picked is not None or read_alone):
                with self._cache_lock:
                    cache[(stamp.key, dcid)] = picked
        known = {**fetched, **hits}
        return {dcid: value for dcid in unique_dcids if (value := known.get(dcid)) is not None}

    def _fetch_single_arc_chunk(
        self,
        dcids: list[str],
        *,
        property_arc: str,
        semaphore: threading.BoundedSemaphore | None,
        deadline: float | None = None,
    ) -> dict:
        """Read one `/v2/node` page. This mixer repeats continuation tokens for this request shape,
        so discard the whole response and halve the dcid chunk into fresh requests. A continuation
        for one dcid raises the read error. A front-door refusal halves the chunk the same way
        (`_read_refused_halves`).

        Acquire `semaphore` for this POST and release it before recursive retries. `deadline`
        bounds the wait and raises `_ChunkDeadlineExpired` when time expires. A bound request
        deadline (`dfo_platform.request_deadline`) also bounds the wait and raises the configured
        deadline error when it is the one that expires.
        """
        acquired = False
        if semaphore is not None:
            wait_deadline = bounded_deadline(deadline)
            timeout = None if wait_deadline is None else max(0.0, wait_deadline - time.monotonic())
            acquired = semaphore.acquire(timeout=timeout)
            if not acquired:
                detail = (
                    f"deadline elapsed before {len(dcids)} dcid(s) reached the front of "
                    "the node-arc semaphore queue"
                )
                if deadline is not None and time.monotonic() >= deadline:
                    raise _ChunkDeadlineExpired(detail)
                raise self._deadline_error_cls(f"{_DEADLINE_EXCEEDED_DETAIL}: {detail}")
        try:
            if deadline is not None and time.monotonic() >= deadline:
                raise _ChunkDeadlineExpired(
                    f"deadline elapsed before {len(dcids)} dcid(s) reached the front of "
                    "the node-arc semaphore queue"
                )
            body = self._post_node({"nodes": dcids, "property": property_arc}, deadline=deadline)
        finally:
            if semaphore is not None and acquired:
                semaphore.release()
        if body is None:
            return self._read_refused_halves(
                dcids,
                lambda part: self._fetch_single_arc_chunk(
                    list(part), property_arc=property_arc, semaphore=semaphore, deadline=deadline
                ),
            )
        if not isinstance(body, dict):
            raise self._read_error_cls("graph /v2/node returned a malformed response body")
        data = body.get("data", {})
        if not isinstance(data, dict):
            raise self._read_error_cls("graph /v2/node returned a malformed data mapping")
        for dcid in dcids:
            if dcid not in data:
                continue
            node_data = data[dcid]
            if not isinstance(node_data, dict):
                raise self._read_error_cls(
                    f"graph /v2/node returned malformed node data for {dcid!r}"
                )
            if "arcs" in node_data and not isinstance(node_data["arcs"], dict):
                raise self._read_error_cls(f"graph /v2/node returned malformed arcs for {dcid!r}")
        if body.get("nextToken") is None:
            return data
        if len(dcids) == 1:
            raise self._read_error_cls(
                f"graph /v2/node reported a continuation reading {property_arc} for "
                f"{dcids[0]!r} alone; refusing to return a short mapping"
            )
        mid = len(dcids) // 2
        left = self._fetch_single_arc_chunk(
            dcids[:mid], property_arc=property_arc, semaphore=semaphore, deadline=deadline
        )
        right = self._fetch_single_arc_chunk(
            dcids[mid:], property_arc=property_arc, semaphore=semaphore, deadline=deadline
        )
        return {**left, **right}

    def _fetch_arc_chunks(
        self,
        dcids: list[str],
        *,
        property_arc: str,
        chunk_size: int,
        semaphore: threading.BoundedSemaphore | None,
        deadline: float | None = None,
    ) -> dict:
        """Fetch `property_arc` in chunks of at most `chunk_size`. Multiple chunks run in a
        per-call pool sized by `node_arc_call_concurrency`, each holding a `semaphore` slot for
        its POST when one is given. A failure cancels pending futures. One absolute `deadline`
        bounds every chunk's semaphore wait and request.
        """
        chunks = list(_chunked(dcids, chunk_size))
        if len(chunks) <= 1:
            return self._fetch_single_arc_chunk(
                dcids, property_arc=property_arc, semaphore=semaphore, deadline=deadline
            )

        def _read_chunk(chunk: list[str]) -> dict:
            return self._fetch_single_arc_chunk(
                chunk, property_arc=property_arc, semaphore=semaphore, deadline=deadline
            )

        merged: dict = {}
        with ThreadPoolExecutor(
            max_workers=min(len(chunks), self._node_arc_call_concurrency)
        ) as pool:
            futures = [submit_with_context(pool, _read_chunk, chunk) for chunk in chunks]
            try:
                for future in as_completed(futures):
                    merged.update(future.result())
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        return merged

    def node_arcs(self, dcid: str) -> dict | None:
        """Return the full `->*` arcs dict, or `None` for an absent node. Delegates to
        `node_arcs_batch`, so both entry points share one code path and cache.
        """
        return self.node_arcs_batch([dcid]).get(dcid)

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        """Fetch each dcid's `->*` arcs in chunks of `node_arc_batch_chunk_size`. This mixer repeats
        continuation tokens for this shape, so chunks halve on continuation. An absent or empty
        result is retried once as a group, except for a single-dcid request. Missing retry results
        raise the configured read error. A node the graph answered with empty arcs when read alone
        (a single-dcid request or a single-suspect retry) caches as `None` under the stamp; an empty
        answer from a multi-dcid group retry is returned uncached, because a `->*` batch can empty
        an entry depending on which other dcids share it. A dcid the response omitted caches
        nothing. Results are deduplicated in
        first-occurrence order. They share the optional cache with `node_arcs`. One absolute
        `deadline` bounds every request and semaphore wait; past it the configured deadline error
        raises.
        """
        if not dcids:
            return {}
        stamp = self._generation.stamp()
        unique_dcids = list(dict.fromkeys(dcids))
        result: dict[str, dict | None] = {}
        misses: list[str] = []
        if self._arcs_cache is not None:
            with self._cache_lock:
                for dcid in unique_dcids:
                    cache_key = (stamp.key, dcid)
                    if cache_key in self._arcs_cache:
                        self._arcs_counters.hits += 1
                        result[dcid] = self._arcs_cache[cache_key]
                    else:
                        self._arcs_counters.misses += 1
                        misses.append(dcid)
        else:
            misses = unique_dcids
            self._arcs_counters.misses += len(misses)

        if not misses:
            return result

        data = self._post_node_arcs_batch(misses, deadline=deadline)
        suspects: list[str] = []
        single_batch = len(misses) == 1
        for dcid in misses:
            present = data.get(dcid) is not None
            arcs = _dedup_wire_arcs(data[dcid].get("arcs", {})) if present else {}
            if not arcs and not (present and single_batch):
                suspects.append(dcid)
                continue
            result[dcid] = arcs or None
            self._remember_arcs(stamp, dcid, result[dcid])

        if suspects:
            retry_data = self._post_node_arcs_batch(suspects, deadline=deadline)
            unanswered = [dcid for dcid in suspects if dcid not in retry_data]
            if unanswered:
                omitted = GraphOmittedDcidsError(
                    f"graph /v2/node omitted {unanswered!r} from both the initial batch "
                    "and a single-dcid retry",
                    dcids=unanswered,
                )
                raise self._read_error_cls(str(omitted)) from omitted
            read_alone = len(suspects) == 1
            for dcid in suspects:
                arcs = (
                    _dedup_wire_arcs(retry_data[dcid].get("arcs", {}))
                    if retry_data[dcid] is not None
                    else {}
                )
                result[dcid] = arcs or None
                # A group retry can empty several suspects together for the same composition
                # reason as the first batch, so only a dcid read alone is trusted empty.
                if arcs or read_alone:
                    self._remember_arcs(stamp, dcid, result[dcid])

        return result

    def _remember_arcs(self, stamp: GenerationStamp, dcid: str, arcs: dict | None) -> None:
        """Cache a read the graph answered: the arcs, or `None` for a node it confirmed empty."""
        if self._arcs_cache is not None:
            with self._cache_lock:
                self._arcs_cache[(stamp.key, dcid)] = arcs

    def _post_node_arcs_batch(self, dcids: list[str], *, deadline: float | None) -> dict:
        try:
            return self._fetch_arc_chunks(
                dcids,
                property_arc="->*",
                chunk_size=self._node_arc_batch_chunk_size,
                semaphore=self._node_arc_semaphore,
                deadline=deadline,
            )
        except _ChunkDeadlineExpired as exc:
            raise self._deadline_error_cls(f"{_DEADLINE_EXCEEDED_DETAIL}: {exc}") from exc

    def exists(self, dcid: str) -> bool:
        """`True` if the node has any arcs. Duck-types `search.eval.graph.
        GraphClient`."""
        return self.node_arcs(dcid) is not None

    def exists_batch(self, dcids: list[str]) -> set[str]:
        """Return dcids with arc evidence. Batch-fetch labels first, then probe unlabeled nodes with
        `->*` before classifying them as absent.
        """
        unique_dcids = list(dict.fromkeys(dcids))
        if not unique_dcids:
            return set()
        known = set(self.node_labels_batch(unique_dcids))
        for dcid in unique_dcids:
            if dcid not in known and self.node_arcs(dcid) is not None:
                known.add(dcid)
        return known

    def node_arc_values(
        self, dcid: str, *, properties: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        """Read every requested property in one `/v2/node` call. Return distinct values in
        first-occurrence order and `()` for an absent arc. Empty `properties` returns `{}`. This
        caller-specified single-node read is uncached.
        """
        if not properties:
            return {}
        body = self.post(NODE_PATH, {"nodes": [dcid], "property": _arc_spec(properties)})
        node_data = body.get("data", {}).get(dcid) or {}
        arcs = node_data.get("arcs", {})
        return parse_arcs(arcs, properties)

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Return every requested dcid. Missing graph nodes map to `None`. Held nodes map each
        property to distinct values in first-occurrence order, using `()` for an absent arc.

        The mixer caps rows globally and paginates with top-level `nextToken`. Pages merge by dcid
        and property. Repeated tokens or a token beyond `_NODE_ARCS_MANY_MAX_PAGES` raise the read
        error. Initial reads use `node_arc_chunk_size`, `node_arc_call_concurrency` and
        `node_arc_many_total_concurrency`.

        Empty results from multi-dcid batches get one group retry, then individual probes up to
        `individual_probe_budget` (`None` probes all). The budget covers the whole call. Missing
        retry or probe entries raise the read error. Unprobed entries map to `None` and are counted
        in one log message. A single-dcid empty result is trusted.

        One absolute `deadline` bounds every request. An entry without `dcid` or `value` contributes
        `UNRESOLVED_NODE`, distinct from an absent arc. This method is uncached.
        """
        if not dcids:
            return {}
        return self._node_arcs_many_batch(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Return deduplicated `name` then `alternateName` values in first-occurrence order. Missing
        nodes and nodes without either arc map to `()`. Filter `UNRESOLVED_NODE`. `deadline` follows
        `node_arcs_many`.
        """
        if not dcids:
            return {}
        arcs = self.node_arcs_many(dcids, properties=_NAME_PROPERTIES, deadline=deadline)
        return {
            dcid: tuple(
                dict.fromkeys(
                    value
                    for prop in _NAME_PROPERTIES
                    for value in (arcs.get(dcid) or {}).get(prop, ())
                    if value != UNRESOLVED_NODE
                )
            )
            for dcid in dict.fromkeys(dcids)
        }

    def _node_arcs_many_batch(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None,
        deadline: float | None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        data = self._post_node_arcs_chunked(dcids, properties=properties, deadline=deadline)
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        suspects: list[str] = []
        single_dcid = len(dcids) == 1
        for dcid in dcids:
            present = dcid in data
            arcs = (data.get(dcid) or {}).get("arcs", {})
            if not arcs:
                if present and single_dcid:
                    result[dcid] = None
                    continue
                suspects.append(dcid)
                continue
            result[dcid] = parse_arcs(arcs, properties, mark_unresolved=True)

        if suspects:
            retry_data = self._post_node_arcs(suspects, properties=properties, deadline=deadline)
            still_missing = [dcid for dcid in suspects if dcid not in retry_data]
            if still_missing:
                omitted = GraphOmittedDcidsError(
                    f"graph omitted {still_missing!r} from /v2/node data even after a "
                    "re-request scoped to just those dcids",
                    dcids=still_missing,
                )
                raise self._read_error_cls(str(omitted)) from omitted
            still_empty: list[str] = []
            for dcid in suspects:
                retry_arcs = (retry_data.get(dcid) or {}).get("arcs", {})
                if retry_arcs:
                    result[dcid] = parse_arcs(retry_arcs, properties, mark_unresolved=True)
                else:
                    still_empty.append(dcid)

            if individual_probe_budget is None:
                to_probe, unprobed = still_empty, []
            else:
                to_probe = still_empty[:individual_probe_budget]
                unprobed = still_empty[individual_probe_budget:]

            for dcid in to_probe:
                probe_data = self._post_node_arcs([dcid], properties=properties, deadline=deadline)
                if dcid not in probe_data:
                    omitted = GraphOmittedDcidsError(
                        f"graph omitted {dcid!r} from /v2/node data even after an "
                        "individual re-request scoped to just that dcid",
                        dcids=[dcid],
                    )
                    raise self._read_error_cls(str(omitted)) from omitted
                probe_arcs = (probe_data.get(dcid) or {}).get("arcs", {})
                result[dcid] = (
                    parse_arcs(probe_arcs, properties, mark_unresolved=True) if probe_arcs else None
                )

            for dcid in unprobed:
                result[dcid] = None
            if unprobed:
                logger.info(
                    "node_arcs_many: %d dcid(s) left unprobed beyond individual_probe_budget=%d",
                    len(unprobed),
                    individual_probe_budget,
                )

        return result

    def _post_node_arcs_chunked(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None
    ) -> dict:
        chunks = list(_chunked(dcids, self._node_arc_chunk_size))
        if len(chunks) <= 1:
            return self._post_node_arcs(dcids, properties=properties, deadline=deadline)

        def _read_chunk(chunk: Sequence[str]) -> dict:
            return self._post_node_arcs(chunk, properties=properties, deadline=deadline)

        merged: dict = {}
        with ThreadPoolExecutor(
            max_workers=min(len(chunks), self._node_arc_call_concurrency)
        ) as pool:
            futures = [submit_with_context(pool, _read_chunk, chunk) for chunk in chunks]
            try:
                for future in as_completed(futures):
                    merged.update(future.result())
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        return merged

    def _post_node_arcs(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> dict:
        """Page the arcs of `dcids` until exhaustion. A front-door refusal on any page reads the
        dcids again as two halves (`_read_refused_halves`)."""
        merged: dict[str, dict] = {}
        token: str | None = None
        for _ in range(_NODE_ARCS_MANY_MAX_PAGES):
            payload: dict = {"nodes": list(dcids), "property": _arc_spec(properties)}
            if token is not None:
                payload["nextToken"] = token
            self._acquire_node_arc_slot(deadline)
            try:
                body = self._post_node(payload, deadline=deadline)
            finally:
                self._node_arc_many_semaphore.release()
            if body is None:
                return self._read_refused_halves(
                    dcids,
                    lambda part: self._post_node_arcs(
                        part, properties=properties, deadline=deadline
                    ),
                )
            _merge_node_arcs_page(merged, body.get("data", {}))
            next_token = body.get("nextToken")
            if next_token is not None and next_token == token:
                stuck = GraphNotAdvancingError(
                    f"graph repeated the same nextToken paging arcs for {list(dcids)!r} "
                    "instead of making progress",
                    dcid=", ".join(dcids),
                    property=payload["property"],
                )
                raise self._read_error_cls(str(stuck)) from stuck
            token = next_token
            if token is None:
                return merged
        raise self._read_error_cls(
            f"graph still reported a nextToken after {_NODE_ARCS_MANY_MAX_PAGES} pages "
            f"reading arcs for {list(dcids)!r}"
        )

    def _acquire_node_arc_slot(self, deadline: float | None) -> None:
        """Acquire a `node_arcs_many` slot, with `deadline` and a bound request deadline
        (`dfo_platform.request_deadline`) bounding the wait."""
        deadline = bounded_deadline(deadline)
        if deadline is None:
            self._node_arc_many_semaphore.acquire()
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._node_arc_many_semaphore.acquire(timeout=remaining):
            raise self._deadline_error_cls(_DEADLINE_EXCEEDED_DETAIL)

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Resolve all names in one `/v2/resolve` request using the caller's property expression.
        Return distinct candidate dcids in mixer order. Names with no candidates map to `[]`. Empty
        input returns `{}`.
        """
        if not names:
            return {}
        body = self.post(_RESOLVE_PATH, {"nodes": list(names), "property": arc}, deadline=deadline)
        by_name: dict[str, list[str]] = {name: [] for name in names}
        for entry in body.get("entities", []):
            node = entry.get("node")
            if node not in by_name:
                continue
            by_name[node] = list(
                dict.fromkeys(
                    dcid for c in entry.get("candidates", []) if (dcid := c.get("dcid")) is not None
                )
            )
        return by_name

    def child_places_batch(
        self,
        parent_dcids: Sequence[str],
        child_type: str,
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        """Query `<-containedInPlace+{typeOf:<child_type>}` for each parent. The mixer times out
        when one request names two or more non-trivial parents, so parents are fetched separately
        and in order. `child_type` accepts positional or keyword form.

        Empty results map to `[]`. Pages follow top-level `nextToken` up to
        `_NODE_ARCS_MANY_MAX_PAGES`. A non-empty `Earth`/`Country` result is memoised when
        `single_value_cache_enabled` is true. An empty one is read again on the next call: Earth
        holds every country, so a listing naming none is a failed read. Other calls are uncached.
        """
        if not parent_dcids:
            return {}
        stamp = self._generation.stamp()
        is_earth_countries = (
            self._single_value_cache_enabled
            and list(parent_dcids) == ["Earth"]
            and child_type == "Country"
        )
        if is_earth_countries:
            with self._cache_lock:
                cached = self._earth_countries_cache
                if (
                    cached is not None
                    and cached[0].key == stamp.key
                    and self._timer() - cached[1] < cached[0].max_age_s
                ):
                    self._earth_countries_counters.hits += 1
                    return {"Earth": list(cached[2])}
                self._earth_countries_counters.misses += 1
        reject_unsafe_filter_token(child_type)
        prop = f"<-containedInPlace+{{typeOf:{child_type}}}"
        result = {
            parent: self._child_places_for_one_parent(parent, prop, child_type, deadline=deadline)
            for parent in parent_dcids
        }
        if is_earth_countries and result["Earth"]:
            with self._cache_lock:
                self._earth_countries_cache = (stamp, self._timer(), list(result["Earth"]))
        return result

    def _child_places_for_one_parent(
        self, parent: str, prop: str, child_type: str, *, deadline: float | None = None
    ) -> list[str]:
        children: list[str] = []
        seen: set[str] = set()
        token: str | None = None
        for _ in range(_NODE_ARCS_MANY_MAX_PAGES):
            payload: dict = {"nodes": [parent], "property": prop}
            if token is not None:
                payload["nextToken"] = token
            body = self.post(NODE_PATH, payload, deadline=deadline)
            node_data = body.get("data", {}).get(parent)
            if node_data is not None:
                for n in node_data.get("arcs", {}).get("containedInPlace+", {}).get("nodes", []):
                    child_dcid = n.get("dcid")
                    if child_dcid is not None and child_dcid not in seen:
                        seen.add(child_dcid)
                        children.append(child_dcid)
            next_token = body.get("nextToken")
            if next_token is not None and next_token == token:
                stuck = GraphNotAdvancingError(
                    f"graph repeated the same nextToken paging child places for {parent!r} "
                    f"via {child_type!r} instead of making progress",
                    dcid=parent,
                    property=prop,
                )
                raise self._read_error_cls(str(stuck)) from stuck
            token = next_token
            if token is None:
                return children
        raise self._read_error_cls(
            f"graph still reported a nextToken after {_NODE_ARCS_MANY_MAX_PAGES} pages "
            f"paging child places for {parent!r} via {child_type!r}"
        )

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        """Return distinct subjects pointing to `dcid` via `property`, in first-occurrence order.
        Follow `nextToken` until exhaustion or the `(max_results + 1)`th subject confirms
        truncation.
        More than `_NODE_SUBJECTS_MAX_PAGES` pages raises the read error. A repeated token or
        `_NODE_SUBJECTS_STUCK_PAGE_LIMIT` fresh-token pages with no new dcids raises
        `GraphNotAdvancingError`. `filters` narrows subjects by value-bound arcs. One absolute
        `deadline` bounds every page.
        """
        spec = _reverse_arc_spec(property, filters)
        dcids: list[str] = []
        seen: set[str] = set()
        token: str | None = None
        stuck_pages = 0
        for _ in range(_NODE_SUBJECTS_MAX_PAGES):
            payload: dict = {"nodes": [dcid], "property": spec}
            if token is not None:
                payload["nextToken"] = token
            body = self.post(NODE_PATH, payload, deadline=deadline)
            node_data = body.get("data", {}).get(dcid) or {}
            arcs = node_data.get("arcs", {})
            distinct_before = len(dcids)
            for entry in arcs.get(property, {}).get("nodes", []):
                entry_dcid = entry.get("dcid")
                if entry_dcid is not None and entry_dcid not in seen:
                    seen.add(entry_dcid)
                    dcids.append(entry_dcid)
            next_token = body.get("nextToken")
            if next_token is not None and next_token == token:
                raise GraphNotAdvancingError(
                    f"graph repeated the same nextToken paging subjects for {dcid!r} via "
                    f"{property!r} instead of making progress",
                    dcid=dcid,
                    property=property,
                )
            if next_token is not None and len(dcids) == distinct_before:
                stuck_pages += 1
                if stuck_pages >= _NODE_SUBJECTS_STUCK_PAGE_LIMIT:
                    raise GraphNotAdvancingError(
                        f"graph reported a fresh nextToken paging subjects for {dcid!r} via "
                        f"{property!r} without contributing a new distinct dcid across "
                        f"{stuck_pages} consecutive pages",
                        dcid=dcid,
                        property=property,
                    )
            else:
                stuck_pages = 0
            token = next_token
            if len(dcids) > max_results or token is None:
                break
        else:
            raise self._read_error_cls(
                f"graph still reported a nextToken after {_NODE_SUBJECTS_MAX_PAGES} pages "
                f"paging subjects for {dcid!r} via {property!r}"
            )
        if len(dcids) > max_results:
            return SubjectPage(dcids=tuple(dcids[:max_results]), truncated=True)
        return SubjectPage(dcids=tuple(dcids), truncated=token is not None)

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, tuple[tuple[str, ...], SubjectsCoverageFlag]]:
        """Read incoming `<-{property}` subjects for each lead dcid in one or more `/v2/node` POSTs.
        Return every input dcid with `(subjects, coverage)`. Subjects are distinct, ordered, capped
        at `max_results`, and deduplicated across provenances. A missing lead or empty reverse arc
        returns `((), "complete")`. An extra confirmed subject gives `"capped"`.

        Pages share one `nextToken` walk. Capped leads leave later page requests while the rest
        continue. A repeated token raises the read error. After `_NODE_SUBJECTS_MANY_MAX_PAGES`,
        uncapped leads return `"budget_exhausted"`. Capped leads retain `"capped"`.

        Empty reverse arcs are ordinary leaf results. One absolute `deadline` bounds every page.
        """
        if not dcids:
            return {}
        unique_dcids = list(dict.fromkeys(dcids))
        reject_unsafe_property_token(property)
        arc_spec = f"<-{property}"
        seen: dict[str, set[str]] = {dcid: set() for dcid in unique_dcids}
        collected: dict[str, list[str]] = {dcid: [] for dcid in unique_dcids}
        capped: set[str] = set()
        token: str | None = None
        for _ in range(_NODE_SUBJECTS_MANY_MAX_PAGES):
            pending = [dcid for dcid in unique_dcids if dcid not in capped]
            payload: dict = {"nodes": pending, "property": arc_spec}
            if token is not None:
                payload["nextToken"] = token
            body = self.post(NODE_PATH, payload, deadline=deadline)
            data = body.get("data", {})
            for dcid in pending:
                node_data = data.get(dcid) or {}
                for entry in node_data.get("arcs", {}).get(property, {}).get("nodes", []):
                    subject_dcid = entry.get("dcid")
                    if subject_dcid is None or subject_dcid in seen[dcid]:
                        continue
                    seen[dcid].add(subject_dcid)
                    collected[dcid].append(subject_dcid)
                    if len(collected[dcid]) > max_results:
                        capped.add(dcid)
                        break
            next_token = body.get("nextToken")
            if next_token is not None and next_token == token:
                stuck = GraphNotAdvancingError(
                    f"graph repeated the same nextToken paging subjects for {unique_dcids!r} "
                    f"via {property!r} instead of making progress",
                    dcid=", ".join(unique_dcids),
                    property=arc_spec,
                )
                raise self._read_error_cls(str(stuck)) from stuck
            token = next_token
            if token is None or len(capped) == len(unique_dcids):
                return {
                    dcid: (
                        (tuple(collected[dcid][:max_results]), "capped")
                        if dcid in capped
                        else (tuple(collected[dcid]), "complete")
                    )
                    for dcid in unique_dcids
                }
        return {
            dcid: (
                (tuple(collected[dcid][:max_results]), "capped")
                if dcid in capped
                else (tuple(collected[dcid]), "budget_exhausted")
            )
            for dcid in unique_dcids
        }

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        """Fetch `->observationProperties` in chunks of `node_arc_batch_chunk_size`, halving on
        continuation. Return distinct targets in first-occurrence order. Absent arcs and nodes map
        to `()`. An empty response is trusted, while an omitted dcid gets one retry.

        A `SingleFlightCache` stores results with capacity at least 1 and coalesces same-generation
        requests. Misses are read in a shared flight whose deadline is
        `observation_properties_flight_timeout_s` from when it starts, whichever caller started
        it. The flight stops scheduling chunks at that deadline and raises `TimeoutError` to every
        caller still waiting. An in-flight chunk finishes and its result is dropped.

        Each caller, the one that started the flight included, stops waiting at the earlier of
        its `declared_role_refresh_deadline` and its bound request deadline, while the flight runs
        on and publishes. A passed refresh deadline raises `TimeoutError`; otherwise the passed
        request deadline raises the configured deadline error, as every other read does. An error
        the flight raises reaches every waiter unchanged.
        """
        if not dcids:
            return {}
        refresh_deadline = _declared_role_refresh_deadline.get()
        try:
            return self._observation_properties_flight.get_or_fetch_many(
                dcids, self._fetch_observation_properties_many, deadline=refresh_deadline
            )
        except WaitTimeout as exc:
            if refresh_deadline is not None and time.monotonic() >= refresh_deadline:
                raise
            raise self._deadline_error_cls(
                f"{_DEADLINE_EXCEEDED_DETAIL}: waiting on the shared observation-property read"
            ) from exc

    def _fetch_observation_properties_many(
        self, dcids: list[str], deadline: float
    ) -> dict[str, tuple[str, ...]]:
        """Fetch owned misses by the flight's `deadline` and parse target tuples. Raise the read
        error for dcids missing from both the batch and the single-dcid retry.
        """
        data = self._fetch_observation_property_chunks(dcids, deadline=deadline)
        absent = [dcid for dcid in dcids if dcid not in data]
        if absent:
            retry_data = self._fetch_observation_property_chunks(absent, deadline=deadline)
            unanswered = [dcid for dcid in absent if dcid not in retry_data]
            if unanswered:
                omitted = GraphOmittedDcidsError(
                    f"graph omitted {unanswered!r} from both the initial batch and "
                    "a single-dcid retry",
                    dcids=unanswered,
                )
                raise self._read_error_cls(str(omitted)) from omitted
            data = {**data, **retry_data}
        result: dict[str, tuple[str, ...]] = {}
        for dcid in dcids:
            node_data = data.get(dcid)
            arcs = node_data.get("arcs", {}) if node_data is not None else {}
            prop_nodes = _dedup_wire_arcs(arcs).get("observationProperties", {}).get("nodes", [])
            result[dcid] = tuple(n["dcid"] for n in prop_nodes if n.get("dcid"))
        return result

    def _fetch_observation_property_chunks(self, dcids: list[str], *, deadline: float) -> dict:
        chunks = list(_chunked(dcids, self._node_arc_batch_chunk_size))
        if not chunks:
            return {}

        def _expired() -> bool:
            return time.monotonic() >= deadline

        def _read_chunk(chunk: list[str]) -> dict:
            return self._fetch_single_arc_chunk(
                chunk,
                property_arc="->observationProperties",
                semaphore=self._node_arc_semaphore,
                deadline=deadline,
            )

        if len(chunks) == 1:
            if _expired():
                raise TimeoutError(
                    "node_observation_properties_batch: the shared read's deadline "
                    "elapsed before its one chunk could be read"
                )
            try:
                return _read_chunk(dcids)
            except _ChunkDeadlineExpired as exc:
                raise TimeoutError(
                    "node_observation_properties_batch: the shared read's deadline "
                    "elapsed while its one chunk queued for the node-arc semaphore"
                ) from exc

        max_workers = min(len(chunks), self._node_arc_call_concurrency)
        merged: dict = {}
        chunk_iter = iter(chunks)
        budget_exhausted = False
        in_flight: dict[Future, list[str]] = {}
        pool = ThreadPoolExecutor(max_workers=max_workers)

        def _submit_more() -> None:
            nonlocal budget_exhausted
            while len(in_flight) < max_workers:
                if _expired():
                    budget_exhausted = True
                    return
                chunk = next(chunk_iter, None)
                if chunk is None:
                    return
                in_flight[submit_with_context(pool, _read_chunk, chunk)] = chunk

        try:
            _submit_more()
            while in_flight and not budget_exhausted:
                remaining = max(0.0, deadline - time.monotonic())
                done, _pending = wait(
                    list(in_flight), timeout=remaining, return_when=FIRST_COMPLETED
                )
                if not done:
                    budget_exhausted = True
                    break
                for future in done:
                    del in_flight[future]
                    try:
                        merged.update(future.result())
                    except _ChunkDeadlineExpired:
                        budget_exhausted = True
                if not budget_exhausted:
                    _submit_more()
        except Exception:
            for future in in_flight:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=not budget_exhausted, cancel_futures=budget_exhausted)

        if budget_exhausted:
            raise TimeoutError(
                "node_observation_properties_batch: the shared read's deadline elapsed "
                "with chunks left unread"
            )
        return merged

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        """One pair's slice of `observation_facets_batch`."""
        return (
            self.observation_facets_batch([stat_var], [entity], needs_dates)
            .get(stat_var, {})
            .get(entity, [])
        )

    def observation_facets_batch(
        self,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        domains: tuple[str, ...] | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        """Return facets for every requested StatVar and entity pair, through the optional
        observation cache keyed `(stamp, stat_var, entity, needs_dates, domains)`. `needs_dates`
        adds per-observation dates and values. `domains` filters facets by provenance hostname;
        `None` sends no filter. Missing pairs map to `[]`. Empty input returns `{}`.

        One read covers the StatVars and entities of the missed pairs. `max_bytes` caps that
        read's response bytes and raises the cap error while reading, and `deadline` bounds it
        as `post` does. A cache hit is returned whatever the cap and deadline.
        """
        if not stat_vars or not entities:
            return {}
        stamp = self._generation.stamp()
        result: dict[str, dict[str, list[Facet]]] = {sv: {} for sv in stat_vars}
        miss_pairs: list[tuple[str, str]] = []
        for sv in stat_vars:
            for entity in entities:
                cache_key = (stamp.key, sv, entity, needs_dates, domains)
                if self._obs_cache is not None:
                    with self._cache_lock:
                        if cache_key in self._obs_cache:
                            self._obs_counters.hits += 1
                            result[sv][entity] = self._obs_cache[cache_key]
                            continue
                        self._obs_counters.misses += 1
                result[sv][entity] = []
                miss_pairs.append((sv, entity))
        if not miss_pairs:
            return result

        miss_svs = list(dict.fromkeys(sv for sv, _ in miss_pairs))
        miss_entities = list(dict.fromkeys(entity for _, entity in miss_pairs))
        read = self._read_observation_facets(
            miss_svs,
            miss_entities,
            with_dates=needs_dates,
            max_bytes=max_bytes,
            deadline=deadline,
            domains=domains,
        )
        for sv in miss_svs:
            for entity in miss_entities:
                cache_key = (stamp.key, sv, entity, needs_dates, domains)
                facets = read[sv][entity]
                if self._obs_cache is not None:
                    with self._cache_lock:
                        if cache_key in self._obs_cache:
                            result[sv][entity] = self._obs_cache[cache_key]
                            continue
                        self._obs_cache[cache_key] = facets
                result[sv][entity] = facets
        return result

    def observation_read_groups(
        self,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        *,
        domains: tuple[str, ...] | None = None,
    ) -> list[list[str]]:
        """`stat_vars` split, in order, into the groups an `observation_facets_batch` read of
        them at `entities` sends apart. A POST read names every StatVar in one request, so it
        is one group. A GET read packs the StatVars under the URL budget
        (`_pack_observation_query`), so each group is one request, or one per entity group
        when the entities need packing too. A caller that may stop reading early reads a
        group at a time. Entities the front door refuses are left out, as the read leaves
        them out. Empty `stat_vars` returns `[]`.
        """
        if not stat_vars:
            return []
        budget = self._observation_get_url_budget_bytes
        blocked = self.blocked_observation_entities()
        allowed = [entity for entity in entities if entity not in blocked]
        if budget is None or not allowed:
            return [list(stat_vars)]
        fixed = _observation_get_fixed(
            _facet_select(needs_dates), all_dates=needs_dates, domains=domains
        )
        packed = _pack_observation_query(stat_vars, allowed, fixed=fixed, budget=budget)
        # Every variable group meets every entity group, so each repeats once per entity group.
        return [list(group) for group in dict.fromkeys(tuple(group) for group, _ in packed)]

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        deadline: float | None = None,
        max_bytes: int | None = None,
        domains: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        """Fetch the Cartesian product of StatVars and entities with `select=["variable", "entity",
        "facet"]`, adding `date` and `value` under `needs_dates`. Empty pairs map to `[]`. Empty
        input returns `{}`. This read is uncached. `deadline` follows `post`; `needs_dates`,
        `max_bytes` and `domains` follow `observation_facets_batch`.
        """
        if not stat_vars or not entities:
            return {}
        return self._read_observation_facets(
            stat_vars,
            entities,
            with_dates=needs_dates,
            max_bytes=max_bytes,
            deadline=deadline,
            domains=domains,
        )

    def _read_observation_facets(
        self,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        *,
        with_dates: bool,
        max_bytes: int | None = None,
        deadline: float | None = None,
        domains: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        """Read facets for every pair, uncached, with blocks sharing a `facet_id` merged."""
        body = self._read_observations(
            stat_vars,
            entities,
            select=_facet_select(with_dates),
            all_dates=with_dates,
            max_bytes=max_bytes,
            deadline=deadline,
            domains=domains,
        )
        top_facets_map: dict = body.get("facets", {})
        by_variable = body.get("byVariable", {})
        return {
            stat_var: {
                entity: _parse_ordered_facets(
                    by_variable.get(stat_var, {})
                    .get("byEntity", {})
                    .get(entity, {})
                    .get("orderedFacets", []),
                    top_facets_map,
                    with_dates=with_dates,
                )
                for entity in entities
            }
            for stat_var in stat_vars
        }

    def _read_observations(
        self,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        *,
        select: Sequence[str],
        all_dates: bool = False,
        max_bytes: int | None = None,
        deadline: float | None = None,
        domains: tuple[str, ...] | None = None,
    ) -> dict:
        """Send one `/v2/observation` read and return the mixer body. `all_dates` asks for every
        date, and `domains` sends `filter.domains`. By GET, the body merges each request's
        `byVariable` entries and `facets` map, and `max_bytes` caps the bytes of all requests
        together.
        """
        budget = self._observation_get_url_budget_bytes
        if budget is None:
            payload: dict = {
                "select": list(select),
                "variable": {"dcids": list(stat_vars)},
                "entity": {"dcids": list(entities)},
            }
            if all_dates:
                payload["date"] = ""
            if domains is not None:
                payload["filter"] = {"domains": list(domains)}
            return self.post_capped_bytes(
                _OBSERVATION_PATH, payload, max_bytes=max_bytes, deadline=deadline
            )

        fixed = _observation_get_fixed(select, all_dates=all_dates, domains=domains)
        stamp = self._generation.stamp()
        blocked = self._blocked_observation_entities_for(stamp)
        by_variable: dict[str, dict] = {}
        facets: dict = {}
        read = 0

        def send(var_group: list[str], entity_group: list[str]) -> None:
            """GET one group and merge its body. A front-door refusal naming several entities
            halves them; one naming a single entity records it as blocked and leaves it out."""
            nonlocal read
            params = httpx.QueryParams(
                [
                    *fixed,
                    *(("variable.dcids", dcid) for dcid in var_group),
                    *(("entity.dcids", dcid) for dcid in entity_group),
                ]
            )
            try:
                if max_bytes is None:
                    body = self._request(
                        "GET",
                        _OBSERVATION_PATH,
                        params=params,
                        deadline=deadline,
                        front_door_refusals=True,
                    )
                else:
                    body, size = self._request_capped(
                        "GET",
                        _OBSERVATION_PATH,
                        params=params,
                        max_bytes=max_bytes,
                        already_read=read,
                        deadline=deadline,
                        front_door_refusals=True,
                    )
                    read += size
            except FrontDoorRefused:
                if len(entity_group) == 1:
                    self._record_blocked_observation_entity(stamp, entity_group[0])
                    return
                mid = len(entity_group) // 2
                send(var_group, entity_group[:mid])
                send(var_group, entity_group[mid:])
                return
            for stat_var, entry in body.get("byVariable", {}).items():
                by_entity = by_variable.setdefault(stat_var, {}).setdefault("byEntity", {})
                by_entity.update(entry.get("byEntity", {}))
            facets.update(body.get("facets", {}))

        allowed = [entity for entity in entities if entity not in blocked]
        if allowed:
            for var_group, entity_group in _pack_observation_query(
                stat_vars, allowed, fixed=fixed, budget=budget
            ):
                send(var_group, entity_group)
        return {"byVariable": by_variable, "facets": facets}

    def blocked_observation_entities(self) -> frozenset[str]:
        """Entities an observation GET could not name under the current generation: the front
        door refused a request naming each alone with its place-name refusal
        (`is_front_door_refusal`). Reads leave them out until the generation changes. Always
        empty for POST reads."""
        return self._blocked_observation_entities_for(self._generation.stamp())

    def seed_blocked_observation_entities(self, entities: frozenset[str]) -> None:
        """Adds `entities`, found refused by an earlier probe, to the blocked set under the
        current generation, so no GET names them and no read pays to find them again."""
        if entities:
            self._add_blocked_observation_entities(self._generation.stamp(), entities)

    def _blocked_observation_entities_for(self, stamp: GenerationStamp) -> frozenset[str]:
        with self._cache_lock:
            key, entities = self._blocked_observation_entities
            return entities if key == stamp.key else frozenset()

    def _add_blocked_observation_entities(
        self, stamp: GenerationStamp, entities: frozenset[str]
    ) -> None:
        with self._cache_lock:
            key, held = self._blocked_observation_entities
            if key != stamp.key:
                held = frozenset()
            if not entities <= held:
                self._blocked_observation_entities = (stamp.key, held | entities)

    def _record_blocked_observation_entity(self, stamp: GenerationStamp, entity: str) -> None:
        self._add_blocked_observation_entities(stamp, frozenset({entity}))
        logger.warning(
            "graph observation GET naming %r met the front door's refusal; leaving it out of "
            "reads under "
            "generation %r",
            entity,
            stamp.key,
        )

    def variable_facets_batch(
        self,
        *,
        stat_var: str,
        entities: Sequence[str],
        needs_dates: bool = False,
        deadline: float | None = None,
        domains: tuple[str, ...] | None = None,
    ) -> dict[str, list[Facet]]:
        """One `stat_var`'s slice of `observation_facets_batch_many`."""
        return self.observation_facets_batch_many(
            stat_vars=[stat_var],
            entities=entities,
            needs_dates=needs_dates,
            deadline=deadline,
            domains=domains,
        ).get(stat_var, {})

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        """Always `None`. No declared-default graph convention exists yet."""
        return None

    def un_observation_reader(self) -> None:
        """Always `None`. UN System Data Commons rows are read through
        `origin_catalogue.graph.OriginGraphClient`, which admits them."""
        return None

    def generation_stamp(self) -> GenerationStamp:
        """The stamp this client's caches key on, read from its generation provider without
        I/O."""
        return self._generation.stamp()

    def source_key(self) -> str:
        """The API root this client reads, which identifies its backend."""
        return str(self._client.base_url)

    def un_availability(self) -> None:
        """Always `None`. The UN deployment's availability read sits on
        `origin_catalogue.graph.OriginGraphClient`."""
        return None

    def child_dcids(self, parent_dcid: str) -> list[str]:
        """dcids of nodes whose `->isPartOf` points to `parent_dcid`,
        distinct, first-occurrence order. `[]` when the parent is absent or
        has no children.
        """
        body = self.post(NODE_PATH, {"nodes": [parent_dcid], "property": "<-isPartOf"})
        node_data = body.get("data", {}).get(parent_dcid)
        arcs = (node_data or {}).get("arcs", {}) if node_data is not None else {}
        if not arcs:
            return []
        return list(
            dict.fromkeys(
                n["dcid"] for n in arcs.get("isPartOf", {}).get("nodes", []) if "dcid" in n
            )
        )

    def enum_members(self, property_dcid: str) -> list[str]:
        """Return enum instances from `->rangeIncludes` classes and their `<-typeOf` members. Return
        `[]` when the property has no range classes or the graph confirms no members.
        """
        arcs = self.node_arcs(property_dcid)
        if arcs is None:
            return []
        class_dcids = [
            n["dcid"] for n in arcs.get("rangeIncludes", {}).get("nodes", []) if "dcid" in n
        ]
        if not class_dcids:
            return []
        by_class = self._fetch_typeof_members_paged(class_dcids)
        seen: set[str] = set()
        members: list[str] = []
        for class_dcid in class_dcids:
            for dcid in by_class.get(class_dcid, ()):
                if dcid not in seen:
                    seen.add(dcid)
                    members.append(dcid)
        return members

    def _fetch_typeof_members_paged(self, class_dcids: list[str]) -> dict[str, list[str]]:
        """Page `<-typeOf` members to exhaustion. Repeated tokens or more than
        `_ENUM_MEMBERS_TYPEOF_MAX_PAGES` pages raise the read error. Reverse-arc fan-out requires
        paging.
        """
        seen_per_class: dict[str, set[str]] = {dcid: set() for dcid in class_dcids}
        members: dict[str, list[str]] = {dcid: [] for dcid in class_dcids}
        tokens_seen: set[str] = set()
        token: str | None = None
        for _ in range(_ENUM_MEMBERS_TYPEOF_MAX_PAGES):
            payload: dict = {"nodes": class_dcids, "property": "<-typeOf"}
            if token is not None:
                payload["nextToken"] = token
            body = self.post(NODE_PATH, payload)
            data = body.get("data", {})
            for class_dcid in class_dcids:
                node_data = data.get(class_dcid)
                if node_data is None:
                    continue
                for n in node_data.get("arcs", {}).get("typeOf", {}).get("nodes", []):
                    dcid = n.get("dcid")
                    if dcid and dcid not in seen_per_class[class_dcid]:
                        seen_per_class[class_dcid].add(dcid)
                        members[class_dcid].append(dcid)
            next_token = body.get("nextToken")
            if next_token is None:
                return members
            if next_token in tokens_seen:
                stuck = GraphNotAdvancingError(
                    f"graph repeated a nextToken paging <-typeOf members for "
                    f"{class_dcids!r} instead of making progress",
                    dcid=", ".join(class_dcids),
                    property="<-typeOf",
                )
                raise self._read_error_cls(str(stuck)) from stuck
            tokens_seen.add(next_token)
            token = next_token
        raise self._read_error_cls(
            f"graph still reported a nextToken after {_ENUM_MEMBERS_TYPEOF_MAX_PAGES} pages "
            f"reading <-typeOf members for {class_dcids!r}"
        )

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        """Map currency unit dcids to ISO 4217 alpha-3 codes, from three sources in rising
        precedence: `CURRENCY_UNIT_OF_MEASURE_ISO_CODES`; the base Data Commons `Currency`
        members, which the graph names by their code (`INR`, `KES`) and gives no
        `iso4217AlphaCode`, each mapped to its own dcid when it is three capital letters; and
        the `CurrencyUnit` members' `->iso4217AlphaCode`. One paged `<-typeOf` read lists both
        classes. Drop and log `CurrencyUnit` members with other than one code. An empty
        `CurrencyUnit` roster raises the read error. Cache one generation-scoped value when
        `single_value_cache_enabled`. An absolute monotonic `deadline` bounds every page, and
        expiry raises the deadline error before the next one; only a complete read is cached.
        """
        stamp = self._generation.stamp()
        if self._single_value_cache_enabled:
            with self._cache_lock:
                cached = self._currency_units_cache
                if (
                    cached is not None
                    and cached[0].key == stamp.key
                    and self._timer() - cached[1] < cached[0].max_age_s
                ):
                    self._currency_units_counters.hits += 1
                    return dict(cached[2])
                self._currency_units_counters.misses += 1

        hop1 = self._post_arc_spec_paged(
            ["CurrencyUnit", _CURRENCY_CLASS], "<-typeOf", deadline=deadline
        )

        def class_members(class_dcid: str) -> list[str]:
            entry = hop1.get(class_dcid) or {}
            nodes = entry.get("arcs", {}).get("typeOf", {}).get("nodes", [])
            return list(dict.fromkeys(dcid for n in nodes if (dcid := n.get("dcid"))))

        member_dcids = class_members("CurrencyUnit")
        named_by_code = {
            dcid: dcid for dcid in class_members(_CURRENCY_CLASS) if _ISO_4217_SHAPE.fullmatch(dcid)
        }

        if not member_dcids:
            raise self._read_error_cls(
                "graph reported no CurrencyUnit members under <-typeOf; a graph with no "
                "currency units is a fault, not an empty roster"
            )

        hop2 = self._post_arc_spec_paged(member_dcids, "->iso4217AlphaCode", deadline=deadline)
        result: dict[str, str] = {**CURRENCY_UNIT_OF_MEASURE_ISO_CODES, **named_by_code}
        for dcid in member_dcids:
            node_entry = hop2.get(dcid)
            values = (
                [
                    v
                    for n in node_entry.get("arcs", {}).get("iso4217AlphaCode", {}).get("nodes", [])
                    if (v := n.get("value"))
                ]
                if node_entry is not None
                else []
            )
            if len(values) != 1:
                logger.warning(
                    "CurrencyUnit %r carries %d iso4217AlphaCode value(s), expected exactly 1; "
                    "dropped from currency_units()",
                    dcid,
                    len(values),
                )
                continue
            result[dcid] = values[0]

        if self._single_value_cache_enabled:
            with self._cache_lock:
                self._currency_units_cache = (stamp, self._timer(), dict(result))
        return result

    def _post_arc_spec_paged(
        self, dcids: Sequence[str], arc_spec: str, *, deadline: float | None = None
    ) -> dict:
        """Page a composed arc expression until exhaustion, merging nodes by dcid and property in
        page order. Repeated tokens or more than `_NODE_ARCS_MANY_MAX_PAGES` pages raise the read
        error. Every page's POST takes the absolute monotonic `deadline`.
        """
        merged: dict[str, dict] = {}
        token: str | None = None
        for _ in range(_NODE_ARCS_MANY_MAX_PAGES):
            payload: dict = {"nodes": list(dcids), "property": arc_spec}
            if token is not None:
                payload["nextToken"] = token
            body = self.post(NODE_PATH, payload, deadline=deadline)
            _merge_node_arcs_page(merged, body.get("data", {}))
            next_token = body.get("nextToken")
            if next_token is not None and next_token == token:
                stuck = GraphNotAdvancingError(
                    f"graph repeated the same nextToken paging {arc_spec!r} for "
                    f"{list(dcids)!r} instead of making progress",
                    dcid=", ".join(dcids),
                    property=arc_spec,
                )
                raise self._read_error_cls(str(stuck)) from stuck
            token = next_token
            if token is None:
                return merged
        raise self._read_error_cls(
            f"graph still reported a nextToken after {_NODE_ARCS_MANY_MAX_PAGES} pages "
            f"reading {arc_spec!r} for {list(dcids)!r}"
        )

    def entities_with_data(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        max_bytes: int | None = None,
    ) -> list[str]:
        """Return entities with data for any requested StatVar. StatVars are chunked by
        `entities_with_data_variable_chunk_size`. Each POST uses all entities and fetching stops
        once every entity is covered. `max_bytes` applies to each response.
        """
        if not variable_dcids or not entity_dcids:
            return []
        wanted = set(entity_dcids)
        present: set[str] = set()
        for i in range(0, len(variable_dcids), self._entities_with_data_variable_chunk_size):
            chunk = variable_dcids[i : i + self._entities_with_data_variable_chunk_size]
            body = self._read_observations(
                chunk, entity_dcids, select=["entity", "variable"], max_bytes=max_bytes
            )
            by_variable = body.get("byVariable", {})
            for var in chunk:
                present |= set(by_variable.get(var, {}).get("byEntity", {}).keys())
            if present >= wanted:
                break
        if not present:
            return []
        return [e for e in entity_dcids if e in present]

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Return entity presence per StatVar in one POST under `deadline`."""
        if not variable_dcids or not entity_dcids:
            return {}
        body = self._read_observations(
            variable_dcids, entity_dcids, select=["entity", "variable"], deadline=deadline
        )
        by_variable = body.get("byVariable", {})
        present = {var: set(by_variable.get(var, {}).get("byEntity", {})) for var in variable_dcids}
        return {var: [e for e in entity_dcids if e in present[var]] for var in variable_dcids}

    def stats(self) -> Mapping[str, int]:
        """Return cumulative process-lifetime hits/misses and current cache entry counts."""
        obs_properties_stats = self._observation_properties_flight.stats()
        with self._cache_lock:
            return {
                "label_hits": self._label_counters.hits,
                "label_misses": self._label_counters.misses,
                "label_entries": 0 if self._label_cache is None else len(self._label_cache),
                "type_hits": self._type_counters.hits,
                "type_misses": self._type_counters.misses,
                "type_entries": 0 if self._type_cache is None else len(self._type_cache),
                "arcs_hits": self._arcs_counters.hits,
                "arcs_misses": self._arcs_counters.misses,
                "arcs_entries": 0 if self._arcs_cache is None else len(self._arcs_cache),
                "obs_hits": self._obs_counters.hits,
                "obs_misses": self._obs_counters.misses,
                "obs_entries": 0 if self._obs_cache is None else len(self._obs_cache),
                "observation_properties_hits": obs_properties_stats["hits"],
                "observation_properties_misses": obs_properties_stats["misses"],
                "observation_properties_entries": obs_properties_stats["entries"],
                "observation_properties_inflight": len(
                    self._observation_properties_flight._sync_inflight
                ),
                "earth_countries_hits": self._earth_countries_counters.hits,
                "earth_countries_misses": self._earth_countries_counters.misses,
                "earth_countries_entries": 0 if self._earth_countries_cache is None else 1,
                "currency_units_hits": self._currency_units_counters.hits,
                "currency_units_misses": self._currency_units_counters.misses,
                "currency_units_entries": 0 if self._currency_units_cache is None else 1,
            }

    def clear(self) -> None:
        """Clear all cached values and observation-properties in-flight bookkeeping."""
        self._observation_properties_flight.clear()
        with self._cache_lock:
            if self._label_cache is not None:
                self._label_cache.clear()
            if self._type_cache is not None:
                self._type_cache.clear()
            if self._arcs_cache is not None:
                self._arcs_cache.clear()
            if self._obs_cache is not None:
                self._obs_cache.clear()
            self._earth_countries_cache = None
            self._currency_units_cache = None
            self._blocked_observation_entities = ("", frozenset())


def _pick_first(values: list[str], dcid: str) -> str | None:
    return values[0] if values else None


def _arc_entries(arcs: dict, prop: str) -> list[dict]:
    """Return the wire entries of `prop` that `dfo_visible` keeps. Both forward-arc parsers read
    through it."""
    return [e for e in arcs.get(prop, {}).get("nodes", []) if dfo_visible(prop, e.get("dcid"))]


def _dedup_wire_arcs(arcs: dict) -> dict:
    """Keep the first arc node for each `dcid` or `value`, preserving order. Keep one unresolved
    entry per property, keyed by `None`. Entries are read through `_arc_entries`.
    """
    deduped: dict = {}
    for prop in arcs:
        seen: set[str | None] = set()
        kept = []
        for entry in _arc_entries(arcs, prop):
            key = entry.get("dcid") or entry.get("value")
            if key in seen:
                continue
            seen.add(key)
            kept.append(entry)
        deduped[prop] = {"nodes": kept}
    return deduped


def _merge_node_arcs_page(merged: dict[str, dict], page_data: dict) -> None:
    """Append each page's nodes to the existing `(dcid, property)` entry in page order."""
    for dcid, node_data in page_data.items():
        node_entry = merged.setdefault(dcid, {"arcs": {}})
        page_arcs = (node_data or {}).get("arcs", {})
        for prop, prop_data in page_arcs.items():
            nodes = node_entry["arcs"].setdefault(prop, {"nodes": []})["nodes"]
            nodes.extend(prop_data.get("nodes", []))


def parse_arcs(
    arcs: dict, properties: Sequence[str], *, mark_unresolved: bool = False
) -> dict[str, tuple[str, ...]]:
    """Return distinct values for each requested property in first-occurrence order. Missing
    properties map to `()`. Entries without `dcid` or `value` are dropped unless
    `mark_unresolved=True`, which inserts `UNRESOLVED_NODE` to distinguish an unresolvable target
    from an absent arc. Entries are read through `_arc_entries`.
    """

    def resolved_value(entry: dict) -> str | None:
        value = entry.get("dcid") or entry.get("value")
        if value is not None:
            return value
        return UNRESOLVED_NODE if mark_unresolved else None

    return {
        prop: tuple(
            dict.fromkeys(
                value
                for entry in _arc_entries(arcs, prop)
                if (value := resolved_value(entry)) is not None
            )
        )
        for prop in properties
    }


def display_name(names: Iterable[str], dcid: str) -> str | None:
    """The one display name of node `dcid` among its `name` values: the greatest by code point,
    `None` where it has none. The graph keys arcs on provenance, so a node can carry several names
    in an order the read does not fix; taking the greatest gives the same name whatever the order.
    A name equal to the dcid, which some imports assert, an empty name, and `UNRESOLVED_NODE` are
    no name."""
    return max(
        (name for name in names if name and name not in (dcid, UNRESOLVED_NODE)), default=None
    )


def label_or_self_name(names: Collection[str], dcid: str) -> str | None:
    """The `display_name` of node `dcid`, or the dcid itself for a node named only by its own
    dcid. `node_label` answers this, and `exists_batch` reads the self-name as evidence the
    node exists."""
    return display_name(names, dcid) or (dcid if dcid in names else None)


def _arc_spec(properties: Sequence[str]) -> str:
    """`"->prop"` for one property, `"->[p1,p2,...]"` for several."""
    for prop in properties:
        reject_unsafe_property_token(prop)
    if len(properties) == 1:
        return f"->{properties[0]}"
    return f"->[{','.join(properties)}]"


def _reverse_arc_spec(property: str, filters: Mapping[str, str] | None) -> str:
    """`"<-prop"` bare, or `"<-prop{k:v,...}"` with `filters` applied,
    sorted by key for a deterministic request."""
    reject_unsafe_property_token(property)
    if not filters:
        return f"<-{property}"
    for key, value in filters.items():
        reject_unsafe_filter_token(key)
        reject_unsafe_filter_token(value)
    pairs = ",".join(f"{key}:{value}" for key, value in sorted(filters.items()))
    return f"<-{property}{{{pairs}}}"


def reject_unsafe_property_token(token: str) -> None:
    """Validate a property token for `->prop`, `<-prop` or `->[p1,p2,...]` syntax. Raise
    `ValueError` for reserved delimiters or whitespace.
    """
    _reject_unsafe_token(token, _UNSAFE_PROPERTY_CHARS)


def reject_unsafe_filter_token(token: str) -> None:
    """Validate a filter key or value for `<-prop{k:v,...}` syntax. Raise `ValueError` for
    reserved delimiters or whitespace.
    """
    _reject_unsafe_token(token, _UNSAFE_FILTER_CHARS)


def _reject_unsafe_token(token: str, unsafe_chars: frozenset[str]) -> None:
    if any(char in unsafe_chars or char.isspace() for char in token):
        raise ValueError(
            f"token {token!r} cannot be safely embedded in the graph's inline expression syntax"
        )


def _chunked(seq: Sequence[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _facet_select(with_dates: bool) -> list[str]:
    """The fields a facet read selects, with each observation's date and value under
    `with_dates`."""
    if with_dates:
        return ["variable", "entity", "date", "value", "facet"]
    return ["variable", "entity", "facet"]


def _observation_get_fixed(
    select: Sequence[str], *, all_dates: bool, domains: tuple[str, ...] | None
) -> list[tuple[str, str]]:
    """The query params every request of one GET observation read carries."""
    fixed = [("select", field) for field in select]
    if all_dates:
        fixed.append(("date", ""))
    if domains is not None:
        fixed.extend(("filter.domains", domain) for domain in domains)
    return fixed


def _pack_observation_query(
    stat_vars: Sequence[str],
    entities: Sequence[str],
    *,
    fixed: Sequence[tuple[str, str]],
    budget: int,
) -> list[tuple[list[str], list[str]]]:
    """Split a GET observation read into `(variables, entities)` groups whose encoded query
    string, with the `fixed` params, fits `budget` bytes. Every requested pair lands in exactly
    one group.

    Both lists go whole when they fit. Otherwise the cheaper list goes whole when it takes at
    most half the room, and the other is packed greedily into the rest. When neither fits in
    half, each is packed into half and every variable group meets every entity group.
    """
    room = budget - len(str(httpx.QueryParams(list(fixed))))
    var_costs = [_query_param_cost("variable.dcids", dcid) for dcid in stat_vars]
    entity_costs = [_query_param_cost("entity.dcids", dcid) for dcid in entities]
    var_total, entity_total = sum(var_costs), sum(entity_costs)
    if var_total + entity_total <= room:
        return [(list(stat_vars), list(entities))]
    if min(var_total, entity_total) <= room // 2:
        if var_total <= entity_total:
            var_groups = [list(stat_vars)]
            entity_groups = _pack_by_cost(entities, entity_costs, room - var_total)
        else:
            var_groups = _pack_by_cost(stat_vars, var_costs, room - entity_total)
            entity_groups = [list(entities)]
    else:
        var_groups = _pack_by_cost(stat_vars, var_costs, room // 2)
        entity_groups = _pack_by_cost(entities, entity_costs, room - room // 2)
    return [(var_group, entity_group) for var_group in var_groups for entity_group in entity_groups]


def _query_param_cost(key: str, value: str) -> int:
    """Encoded bytes one `key=value` pair adds to a query string, with its `&` separator."""
    return len(str(httpx.QueryParams([(key, value)]))) + 1


def _pack_by_cost(dcids: Sequence[str], costs: Sequence[int], room: int) -> list[list[str]]:
    """Pack dcids greedily in input order into groups whose summed cost fits `room`."""
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for dcid, cost in zip(dcids, costs, strict=True):
        if cost > room:
            raise ValueError(
                f"dcid {dcid!r} alone exceeds the observation GET URL budget "
                f"({cost} bytes against {room} bytes of room)"
            )
        if current and used + cost > room:
            groups.append(current)
            current, used = [], 0
        current.append(dcid)
        used += cost
    if current:
        groups.append(current)
    return groups


def _parse_ordered_facets(
    ordered_facets: list[dict], top_facets_map: dict, *, with_dates: bool
) -> list[Facet]:
    """Build one pair's `orderedFacets` blocks and merge those sharing a `facet_id`.
    `with_dates` says whether the read selected dates.
    """
    return _merge_facets_by_id(
        [_build_facet(raw, top_facets_map, with_dates=with_dates) for raw in ordered_facets]
    )


def _build_facet(raw: dict, top_facets_map: dict, *, with_dates: bool) -> Facet:
    """Build one `orderedFacets` block, joining its metadata from the top-level `facets` map.
    Observations without a date are left out of `dates` and `values`.
    """
    facet_id = raw.get("facetId")
    facet_meta = top_facets_map.get(facet_id, {}) if facet_id else {}
    obs_count = raw.get("obsCount", 0)
    dated = [(o["date"], o.get("value")) for o in raw.get("observations", []) if o.get("date")]
    return Facet(
        facet_id=facet_id,
        provenance_id=facet_meta.get("provenanceId"),
        measurement_method=facet_meta.get("measurementMethod"),
        unit=facet_meta.get("unit"),
        scaling_factor=facet_meta.get("scalingFactor"),
        obs_count=obs_count,
        earliest_date=raw.get("earliestDate"),
        latest_date=raw.get("latestDate"),
        observation_period=facet_meta.get("observationPeriod"),
        dates=tuple(date for date, _ in dated),
        values=tuple(value for _, value in dated),
        dates_complete=not with_dates or len(dated) >= obs_count,
    )


def _merge_facets_by_id(facets: list[Facet]) -> list[Facet]:
    """Merge blocks by `facet_id` in first-seen order, summing counts, widening date bounds and
    concatenating dated observations. Keep id-less blocks separate.
    """
    result: list[Facet] = []
    index_by_id: dict[str, int] = {}
    for facet in facets:
        if facet.facet_id is None:
            result.append(facet)
            continue
        existing_index = index_by_id.get(facet.facet_id)
        if existing_index is None:
            index_by_id[facet.facet_id] = len(result)
            result.append(facet)
        else:
            existing = result[existing_index]
            result[existing_index] = replace(
                existing,
                obs_count=existing.obs_count + facet.obs_count,
                earliest_date=earliest_date(existing.earliest_date, facet.earliest_date),
                latest_date=latest_date(existing.latest_date, facet.latest_date),
                dates=existing.dates + facet.dates,
                values=existing.values + facet.values,
                dates_complete=existing.dates_complete and facet.dates_complete,
                annual_evidence=existing.reads_annual and facet.reads_annual,
            )
    return result


def earliest_date(a: str | None, b: str | None) -> str | None:
    """The merge rule for two facets sharing one identity's `earliest_date`."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def latest_date(a: str | None, b: str | None) -> str | None:
    """The merge rule for two facets sharing one identity's `latest_date`."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)
