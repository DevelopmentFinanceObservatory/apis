"""The engine's one call to Jev, TypeSafe's System One model: `ask`.

`ask` sends a state and its questions through one `AsyncTypeSafeClient`, built lazily on
first use and shared by every call in the process. A request too large to send whole goes
as concurrent chunks (`jev_split.split_questions`) whose answers merge into one response.
One `asyncio.timeout` at `Settings.jev_ceiling_s` bounds the whole call: the wait for a
concurrency slot, every chunk, and the one retry. Every failure leaves as `JevError`.

A circuit breaker shared by the process opens after `BREAKER_FAILURES` consecutive failed
calls inside `BREAKER_WINDOW_S`; for `BREAKER_OPEN_S` every call then fails at once without
a request. The first call after that is the probe: its success closes the breaker, its
failure opens it again.
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

if TYPE_CHECKING:
    from search.engine.config import SearchSettings

type JevQuestion = NoulModel | ChoiceModel

type JevErrorKind = Literal[
    "breaker_open",
    "no_key",
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
    the call is the probe; `leave` records its outcome: True for success, False for a Jev
    failure, None for a call abandoned by cancellation, which counts as neither.
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
    """One connection pool per event loop, since a pooled connection belongs to the loop
    that opened it. Pools of closed loops are dropped on the next lookup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ssl_context = httpx2.create_ssl_context()
        self._pools: dict[asyncio.AbstractEventLoop, httpx2.AsyncHTTPTransport] = {}

    def _pool(self) -> httpx2.AsyncHTTPTransport:
        loop = asyncio.get_running_loop()
        with self._lock:
            for closed in [known for known in self._pools if known.is_closed()]:
                del self._pools[closed]
            pool = self._pools.get(loop)
            if pool is None:
                pool = self._pools[loop] = httpx2.AsyncHTTPTransport(
                    verify=self._ssl_context, limits=_LIMITS
                )
            return pool

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return await self._pool().handle_async_request(request)

    async def aclose(self) -> None:
        with self._lock:
            pool = self._pools.pop(asyncio.get_running_loop(), None)
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
_SEMAPHORES: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
_SEMAPHORES_LOCK = threading.Lock()


def _client(settings: SearchSettings) -> AsyncTypeSafeClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = build_client(settings)
        return _CLIENT


def _semaphore(size: int) -> asyncio.Semaphore:
    """The running loop's Jev request semaphore; an asyncio semaphore serves one loop."""
    loop = asyncio.get_running_loop()
    with _SEMAPHORES_LOCK:
        for closed in [known for known in _SEMAPHORES if known.is_closed()]:
            del _SEMAPHORES[closed]
        semaphore = _SEMAPHORES.get(loop)
        if semaphore is None:
            semaphore = _SEMAPHORES[loop] = asyncio.Semaphore(size)
        return semaphore


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


async def _send(
    client: AsyncTypeSafeClient,
    state: JSONContent,
    questions: dict[str, JevQuestion],
    *,
    deadline: float,
    semaphore: asyncio.Semaphore,
) -> SystemOneResponse:
    """One request, retried once on a rate limit or server error when the delay before the
    retry, a `Retry-After` included, ends before `deadline` (loop time)."""
    async with semaphore:
        budget = deadline - asyncio.get_running_loop().time()
        if budget <= 0:
            raise JevError(None, "timeout")
        retry = RetryPolicy(
            max_retries=1,
            http_statuses=set(_RETRY_STATUSES),
            respect_retry_after=True,
            api_connection_error=True,
            api_timeout_error=False,
            timeout=budget,
        )
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
    chunk cancels the rest and fails the call."""
    client = _client(settings)
    semaphore = _semaphore(settings.jev_max_concurrency)
    deadline = asyncio.get_running_loop().time() + settings.jev_ceiling_s
    try:
        async with asyncio.timeout_at(deadline):
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(
                        _send(client, state, chunk, deadline=deadline, semaphore=semaphore)
                    )
                    for chunk in chunks
                ]
    except TimeoutError as exc:
        raise JevError(None, "timeout") from exc
    except ExceptionGroup as group:
        first = group.exceptions[0]
        if isinstance(first, JevError):
            raise first from group
        raise
    return [task.result() for task in tasks]


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
            question came back unanswered.
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
    except JevError:
        ok = False
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
