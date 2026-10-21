"""The engine's one call to Jev, TypeSafe's System One model: `ask`.

`ask` sends a state and its questions through one `AsyncTypeSafeClient`, built lazily on
first use and shared by every call in the process. A request too large to send whole goes
as concurrent chunks (`jev_split.split_questions`) whose answers merge into one response.
One `asyncio.timeout` at `Settings.jev_ceiling_s` bounds the whole call: the wait for a
concurrency slot, every chunk, and the one retry. Every failure leaves as `JevError`. A call
the ceiling cuts fails as ``timeout`` when a request it cut took its slot at once and was
sent, as ``queued_timeout`` when every cut request that was sent had waited for its slot,
and as ``slot_wait`` when no cut request was sent. A ceiling that fires after every request
has answered cuts nothing outstanding, so the call returns those answers.

A circuit breaker shared by the process opens after `BREAKER_FAILURES` consecutive calls
inside `BREAKER_WINDOW_S` that failed in a way that says Jev is unhealthy (`_HEALTH_KINDS`:
a timed-out or failed connection, a 401 or 403, a 402, a 408, a 429 or a 5xx); for
`BREAKER_OPEN_S` every call then fails at once without a request. The first call after that
is the probe: its success closes the breaker, its unhealthy failure opens it again. A
failure particular to the request (a refusal, a missing key, an unanswered question) or a
cut that a wait for a slot shares in (``slot_wait``, ``queued_timeout``) moves the breaker
neither way, since the process's own queue, not Jev, spent the ceiling.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    ChoiceModel,
    JSONContent,
    NoulModel,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeError,
)

from search.engine.binders.jev_split import split_questions
from search.engine.clients.loop_local import LoopLocal

if TYPE_CHECKING:
    from search.engine.config import SearchSettings

type JevQuestion = NoulModel | ChoiceModel

type JevErrorKind = Literal[
    "breaker_open",
    "no_key",
    "slot_wait",
    "queued_timeout",
    "timeout",
    "connection",
    "rate_limited",
    "server",
    "payment_required",
    "request_timeout",
    "auth",
    "rejected",
    "invalid_request",
    "invalid_response",
    "unanswered",
]

BREAKER_FAILURES = 5
BREAKER_WINDOW_S = 10.0
BREAKER_OPEN_S = 30.0

# The failures that say Jev itself is unhealthy, the only ones the breaker counts. A
# rejected key and exhausted credit fail every call on the account, so they count too.
_HEALTH_KINDS: frozenset[JevErrorKind] = frozenset(
    {
        "timeout",
        "request_timeout",
        "connection",
        "rate_limited",
        "server",
        "auth",
        "payment_required",
    }
)

# The one retry goes to a rate limit, a server error, or a connection that failed without
# timing out (a refusal, or a reset on a pooled connection the server closed), which fails
# in milliseconds. A timed-out attempt is not retried: the second attempt would spend what
# is left of the ceiling.
_RETRY_STATUSES = frozenset({429, *range(500, 600)})

# The SDK's own pool limits, which it applies only to a client it builds its transport for.
_LIMITS = httpx2.Limits(max_connections=100, max_keepalive_connections=100, keepalive_expiry=30.0)


class JevError(Exception):
    """A Jev call that returned no usable answer.

    `status` is the HTTP status of the failed response, or None when no response came
    back (a timeout, a connection failure, an open breaker, a missing key). `kind` names
    the failure for logs and metrics.
    """

    def __init__(self, status: int | None, kind: JevErrorKind) -> None:
        super().__init__(f"jev {kind}" if status is None else f"jev {kind} ({status})")
        self.status = status
        self.kind: JevErrorKind = kind


@dataclass(frozen=True, slots=True, kw_only=True)
class JevUsage:
    """Input and output tokens over every request of one call, 0 where Jev reported none."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class JevResponse:
    """Jev's answers to one `ask`, keyed by question id.

    `nouls` holds each noul question's probability of yes, `choices` each choice
    question's probability per label. `model` is the versioned id that answered,
    `seconds` the wall time of the whole call, chunks and retry included, and `chunks` the
    requests the call was split into (`jev_split.split_questions`).
    """

    nouls: dict[str, float]
    choices: dict[str, dict[str, float]]
    model: str
    usage: JevUsage
    seconds: float
    chunks: int


class CircuitBreaker:
    """Consecutive-failure breaker over whole Jev calls, safe across threads.

    `enter` admits a call or raises `JevError(None, "breaker_open")`, and returns whether
    the call is the probe; `leave` records its outcome: True for success, False for a
    failure that says Jev is unhealthy, None for any other failure or a call abandoned by
    cancellation, which counts as neither.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: deque[float] = deque()
        self._open_until: float | None = None
        self._probing = False

    def enter(self) -> bool:
        with self._lock:
            if self._open_until is None:
                return False
            if self._probing or self._clock() < self._open_until:
                raise JevError(None, "breaker_open")
            self._probing = True
            return True

    def leave(self, ok: bool | None, *, probe: bool) -> None:
        with self._lock:
            now = self._clock()
            if probe:
                self._probing = False
                if ok is True:
                    self._open_until = None
                elif ok is False:
                    self._open_until = now + BREAKER_OPEN_S
                return
            if self._open_until is not None or ok is None:
                return
            if ok:
                self._failures.clear()
                return
            self._failures.append(now)
            while self._failures[0] < now - BREAKER_WINDOW_S:
                self._failures.popleft()
            if len(self._failures) >= BREAKER_FAILURES:
                self._failures.clear()
                self._open_until = now + BREAKER_OPEN_S


class _LoopLocalTransport(httpx2.AsyncBaseTransport):
    """One connection pool per event loop (`loop_local.LoopLocal`), since a pooled
    connection belongs to the loop that opened it."""

    def __init__(self) -> None:
        self._ssl_context = httpx2.create_ssl_context()
        self._pools: LoopLocal[httpx2.AsyncHTTPTransport] = LoopLocal()

    def _pool(self) -> httpx2.AsyncHTTPTransport:
        return self._pools.get(
            lambda: httpx2.AsyncHTTPTransport(verify=self._ssl_context, limits=_LIMITS)
        )

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return await self._pool().handle_async_request(request)

    async def aclose(self) -> None:
        pool = self._pools.pop()
        if pool is not None:
            await pool.aclose()


def build_client(
    settings: SearchSettings, *, transport: httpx2.AsyncBaseTransport | None = None
) -> AsyncTypeSafeClient:
    """A System One client on `settings`' model and key, over `transport` when given.

    Compression is zstd by name, so a process without the `zstd` extra fails here rather
    than sending every body uncompressed.

    Raises:
        JevError: The key is empty (`no_key`) or the SDK refuses it (`invalid_request`).
    """
    key = settings.jev_api_key.get_secret_value()
    if not key:
        raise JevError(None, "no_key")
    try:
        return AsyncTypeSafeClient(
            api_key=key,
            model=settings.jev_model,
            timeout=settings.jev_ceiling_s,
            transport=transport or _LoopLocalTransport(),
            compression="zstd",
        )
    except TypeSafeError as exc:
        raise JevError(None, "invalid_request") from exc


_CLIENT: AsyncTypeSafeClient | None = None
_CLIENT_LOCK = threading.Lock()
_BREAKER = CircuitBreaker()
# The running loop's Jev request semaphore; an asyncio semaphore serves one loop.
_SEMAPHORES: LoopLocal[asyncio.Semaphore] = LoopLocal()


def _client(settings: SearchSettings) -> AsyncTypeSafeClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = build_client(settings)
        return _CLIENT


def _jev_error(exc: TypeSafeError) -> JevError:
    """`exc` as a `JevError`. 402 and 408 have no SDK class of their own and are read off
    `.status`."""
    if isinstance(exc, TypeSafeAPITimeoutError):
        return JevError(None, "timeout")
    if isinstance(exc, TypeSafeAPIConnectionError):
        return JevError(None, "connection")
    if isinstance(exc, TypeSafeAPIResponseValidationError):
        return JevError(exc.status, "invalid_response")
    if not isinstance(exc, TypeSafeAPIError):
        return JevError(None, "invalid_request")
    match exc.status:
        case 402:
            kind: JevErrorKind = "payment_required"
        case 408:
            kind = "request_timeout"
        case 429:
            kind = "rate_limited"
        case 401 | 403:
            kind = "auth"
        case status if status >= 500:
            kind = "server"
        case _:
            kind = "rejected"
    return JevError(exc.status, kind)


@dataclass(slots=True)
class _Attempt:
    """One chunk's request: whether every slot was taken when it asked for one (`queued`),
    and whether it holds its slot and started (`sent`)."""

    queued: bool = False
    sent: bool = False


async def _send(
    client: AsyncTypeSafeClient,
    state: JSONContent,
    questions: dict[str, JevQuestion],
    *,
    deadline: float,
    semaphore: asyncio.Semaphore,
    attempt: _Attempt,
) -> SystemOneResponse:
    """One request, retried once on a rate limit or server error when the delay before the
    retry, a `Retry-After` included, ends before `deadline` (loop time), recorded on
    `attempt`."""
    attempt.queued = semaphore.locked()
    async with semaphore:
        budget = deadline - asyncio.get_running_loop().time()
        if budget <= 0:
            raise JevError(None, "slot_wait")
        retry = RetryPolicy(
            max_retries=1,
            http_statuses=set(_RETRY_STATUSES),
            respect_retry_after=True,
            api_connection_error=True,
            api_timeout_error=False,
            timeout=budget,
        )
        attempt.sent = True
        try:
            return await client.system_one(state, questions, retry=retry)
        except TypeSafeError as exc:
            raise _jev_error(exc) from exc


async def _ask_chunks(
    state: JSONContent,
    chunks: list[dict[str, JevQuestion]],
    *,
    settings: SearchSettings,
) -> list[SystemOneResponse]:
    """Every chunk's response, sent concurrently under one deadline. The first failed
    chunk cancels the rest and fails the call. A passed deadline fails it under the kind
    of the requests it cut (`_cut_kind`), unless every chunk had already answered, when it
    returns them: a loop stalled past the deadline can run a response before the deadline's
    timer, and one chunk's answer without the others' completes nothing."""
    client = _client(settings)
    semaphore = _SEMAPHORES.get(lambda: asyncio.Semaphore(settings.jev_max_concurrency))
    attempts = [_Attempt() for _ in chunks]
    deadline = asyncio.get_running_loop().time() + settings.jev_ceiling_s
    tasks: list[asyncio.Task[SystemOneResponse]] = []
    try:
        async with asyncio.timeout_at(deadline):
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(
                        _send(
                            client,
                            state,
                            chunk,
                            deadline=deadline,
                            semaphore=semaphore,
                            attempt=attempt,
                        )
                    )
                    for chunk, attempt in zip(chunks, attempts, strict=True)
                ]
    except TimeoutError as exc:
        cut = [a for a, task in zip(attempts, tasks, strict=True) if not _answered(task)]
        if tasks and not cut:
            return [task.result() for task in tasks]
        raise JevError(None, _cut_kind(cut)) from exc
    except ExceptionGroup as group:
        first = group.exceptions[0]
        if isinstance(first, JevError):
            raise first from group
        raise
    return [task.result() for task in tasks]


def _answered(task: asyncio.Task[SystemOneResponse]) -> bool:
    """Whether `task` ended with a response."""
    return task.done() and not task.cancelled() and task.exception() is None


def _cut_kind(cut: list[_Attempt]) -> JevErrorKind:
    """The kind of a call whose ceiling cut the requests `cut`: ``timeout`` when one of them
    was sent without waiting for its slot, so Jev alone spent the ceiling on it;
    ``queued_timeout`` when each one sent had waited; ``slot_wait`` when none was sent."""
    if any(a.sent and not a.queued for a in cut):
        return "timeout"
    return "queued_timeout" if any(a.sent for a in cut) else "slot_wait"


async def ask(
    state: JSONContent,
    questions: Mapping[str, JevQuestion],
    *,
    settings: SearchSettings,
) -> JevResponse:
    """Jev's answers to `questions` over `state`.

    Raises:
        ValueError: `state`, or `state` with one question, exceeds the request budget.
        JevError: The breaker is open, the call failed or overran `jev_ceiling_s`, or a
            question came back unanswered. Only a kind in `_HEALTH_KINDS` counts against
            the breaker.
    """
    chunks = split_questions(state, questions)
    probe = _BREAKER.enter()
    ok: bool | None = None
    started = time.perf_counter()
    try:
        response = _merge(await _ask_chunks(state, chunks, settings=settings), started=started)
        if not questions.keys() <= response.nouls.keys() | response.choices.keys():
            raise JevError(None, "unanswered")
        ok = True
        return response
    except JevError as exc:
        ok = False if exc.kind in _HEALTH_KINDS else None
        raise
    finally:
        _BREAKER.leave(ok, probe=probe)


def _merge(responses: list[SystemOneResponse], *, started: float) -> JevResponse:
    """One `JevResponse` over every chunk's response, timed from `started`."""
    nouls: dict[str, float] = {}
    choices: dict[str, dict[str, float]] = {}
    for response in responses:
        nouls |= {qid: answer.noul for qid, answer in response.nouls.items()}
        choices |= {qid: dict(answer.probabilities) for qid, answer in response.choices.items()}
    return JevResponse(
        nouls=nouls,
        choices=choices,
        model=responses[0].model,
        usage=JevUsage(
            input_tokens=sum(r.usage.input_tokens or 0 for r in responses),
            output_tokens=sum(r.usage.output_tokens or 0 for r in responses),
        ),
        seconds=time.perf_counter() - started,
        chunks=len(responses),
    )
