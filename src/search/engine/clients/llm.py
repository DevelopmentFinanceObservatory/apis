"""Gemini structured-output client for Search.

``generate_structured`` always calls Gemini. The async engine uses
``generate_structured_async`` to single-flight identical requests when the
LLM has a memo. The client is built lazily. The first live call checks
``GEMINI_API_KEY``.

The default model is the concrete ``gemini-3.5-flash-lite`` id so evaluation
runs use a stable model. Callers can choose a ``-latest`` alias. Model
configuration handles those aliases too.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import json
import logging
import re
import struct
from collections.abc import Callable, Coroutine, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

import search.engine.config as config  # module-attribute access for runtime-resolved values
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from search.engine.clients.loop_local import LoopLocal
from search.engine.errors import LLMInfraError, LLMOutputRefused
from search.engine.response_cache import Lease, ResponseCache

if TYPE_CHECKING:
    from google import genai
    from google.genai.types import GenerateContentConfig, ThinkingConfig

    from search.engine.extraction_store import ExtractionStore

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

logger = logging.getLogger(__name__)

# Lazy singletons, built on first call to generate_structured: the served calls' client, and
# the client of detached calls (`detached`) with its own pools.
_CLIENT = None
_DETACHED_CLIENT = None

# Memo and extraction-store hits report zero usage because they skip Gemini. Live calls return None
# when Gemini omits usage metadata.
MEMO_HIT_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


# The usage key counting the model calls a request sent whose tokens it cannot count.
UNCOUNTED_CALLS = "uncounted_calls"


def sum_usage(a: dict | None, b: dict | None) -> dict | None:
    """Sum two LLM usage dicts key by key, over every key either holds, in the order they
    first appear: the Gemini counts, ``jev_input_tokens`` and ``jev_output_tokens`` from a
    Jev call, and ``uncounted_calls``. Return `None` when both are `None`."""
    if a is None:
        return b
    if b is None:
        return a
    return {key: a.get(key, 0) + b.get(key, 0) for key in a | b}


# Follower wait limit. HttpOptions.timeout is per phase, so a stalled
# connection can outlast it. This overall wait uses the same configured bound.
_LLM_WAITER_DEADLINE_EXCEEDED_DETAIL = (
    "the Gemini structured-output call exceeded SEARCH_LLM_TIMEOUT_S"
)

# Followers receive this generic error when the leader fails.
# The leader's detail belongs to its own caller.
_LLM_MEMO_LEADER_FAILED_DETAIL = "the shared in-flight Gemini structured-output call failed"


class _CallTiming:
    """When the memo-writing call holding this record (`_lead`) left the connection pool's
    queue: the first httpcore trace event of its latest request (`_LoopLocalTransport`).
    The client makes one attempt per call and follows redirects (`_build_client`), each a
    request of its own, so only the request that answered counts. A connection it opens
    counts, its handshake included, since a fresh call pays it too."""

    def __init__(self) -> None:
        self.sent_at: float | None = None

    async def trace(self, _event: str, _info: dict[str, Any]) -> None:
        if self.sent_at is None:
            self.sent_at = asyncio.get_running_loop().time()


_CALL_TIMING: ContextVar[_CallTiming | None] = ContextVar("llm_call_timing", default=None)


class _LoopLocalTransport(httpx.AsyncBaseTransport):
    """Keep one ``max_connections`` pool per event loop.

    Connections belong to their creating loop, so calls on different loops
    need separate pools (`loop_local.LoopLocal`).
    HTTP/1.1 uses one request per connection, so the limit bounds each
    loop's concurrent Gemini requests. Pool waits use the request timeout.
    """

    def __init__(self, *, max_connections: int) -> None:
        self._limits = httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        )
        self._pools: LoopLocal[httpx.AsyncHTTPTransport] = LoopLocal()

    def _pool(self) -> httpx.AsyncHTTPTransport:
        return self._pools.get(lambda: httpx.AsyncHTTPTransport(limits=self._limits))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        timing = _CALL_TIMING.get()
        if timing is not None and request.extensions.get("trace") in (None, timing.trace):
            # httpcore traces a request only once the pool has handed it a connection, so the
            # first event ends the pool wait. A redirect the client follows sends a new
            # request carrying the first one's extensions, and starts the count again.
            timing.sent_at = None
            request.extensions["trace"] = timing.trace
        return await self._pool().handle_async_request(request)

    async def aclose(self) -> None:
        pool = self._pools.pop()
        if pool is not None:
            await pool.aclose()


def _build_client(api_key: str, *, max_connections: int) -> genai.Client:
    """Build a Gemini client using one-attempt httpx requests.

    Passing an async httpx client selects google-genai's httpx path, avoiding
    aiohttp's 1-10 second retry. Passing the client itself also prevents
    google-genai from wrapping the shared transport in extra clients whose
    finalizers can close its loop's pool and exceed the connection cap.

    ``HttpOptions.timeout`` is converted from milliseconds to seconds and
    bounds connect, read, write, and pool waits. Saturated pools raise
    ``httpx.PoolTimeout``, translated to ``LLMInfraError`` by the caller.
    ``max_connections`` caps each event loop's pool.
    """
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError as exc:
        raise LLMInfraError(
            "search.engine requires the 'engine' extra. "
            "Install with: uv sync --extra engine (or pip install 'search[engine]')."
        ) from exc
    async_client = httpx.AsyncClient(
        transport=_LoopLocalTransport(max_connections=max_connections),
        # Every request carries HttpOptions.timeout. google-genai's httpx clients follow redirects.
        timeout=None,
        follow_redirects=True,
    )
    # HttpOptions.timeout uses milliseconds. Convert the seconds-based setting here.
    return genai.Client(
        api_key=api_key,
        http_options=HttpOptions(
            timeout=int(config.settings().llm_timeout_s * 1000),
            httpx_async_client=async_client,
        ),
    )


def _get_client() -> genai.Client:
    """Return (or build) the singleton genai.Client of served calls, with
    ``llm_max_concurrency`` connections per event loop.

    Raises LLMInfraError only when a live call is attempted and GEMINI_API_KEY
    is absent. Tests use FakeLLM instead, bypassing this path.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _build_client(
            _required_api_key(), max_connections=config.settings().llm_max_concurrency
        )
    return _CLIENT


def _get_detached_client() -> genai.Client:
    """Return (or build) the singleton genai.Client of detached calls, with
    ``jev_shadow_max_inflight`` connections per event loop of its own."""
    global _DETACHED_CLIENT
    if _DETACHED_CLIENT is None:
        _DETACHED_CLIENT = _build_client(
            _required_api_key(), max_connections=config.settings().jev_shadow_max_inflight
        )
    return _DETACHED_CLIENT


def _required_api_key() -> str:
    api_key = config.settings().gemini_api_key
    if not api_key:
        raise LLMInfraError(
            "GEMINI_API_KEY is required for live LLM calls. "
            "Set the environment variable or use FakeLLM in tests."
        )
    return api_key


_GEMINI_3_MINOR_RE = re.compile(r"^gemini-3\.(\d+)")

# Shared by temperature and thinking-config selection for Gemini 3.5+ models.
_GEMINI_3_5_PLUS_ALIASES = ("gemini-flash-latest", "gemini-flash-lite-latest")


def _is_gemini_3_5_plus(model: str) -> bool:
    """True for Gemini 3.5, 3.6, and later (and the -latest aliases)."""
    if model in _GEMINI_3_5_PLUS_ALIASES:
        return True
    match = _GEMINI_3_MINOR_RE.match(model)
    return match is not None and int(match.group(1)) >= 5


def _temperature_for_model(model: str) -> float | None:
    """Return the pinned temperature, or ``None`` to omit it.

    Gemini 3.5+ models and their latest aliases disregard temperature,
    ``top_p``, and ``top_k``. Future models may reject them. Older models
    use the pinned ``0.0`` temperature.
    """
    if _is_gemini_3_5_plus(model):
        return None
    return 0.0


# Maps the llm_thinking_level setting's validated lowercase strings to the ThinkingLevel
# enum member name, so _thinking_config_for_model's override branches can getattr
# into the lazily-imported enum without a second mapping at each call site.
_THINKING_LEVEL_NAMES = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}

# Extract keeps the instance's resolved level. Bind, the quantity read, the stated-words read
# and the exclusion read default to minimal. An instance-level or config override takes precedence in
# for_stage().
_STAGE_THINKING_LEVELS: dict[str, str | None] = {
    "extract": None,
    "bind": "minimal",
    "quantity": "minimal",
    "stated_words": "minimal",
    "exclusion_read": "minimal",
}


def _validate_stage_thinking_levels() -> None:
    for stage_name, stage_level in _STAGE_THINKING_LEVELS.items():
        if stage_level is not None and stage_level not in _THINKING_LEVEL_NAMES:
            raise ValueError(
                f"_STAGE_THINKING_LEVELS[{stage_name!r}] = {stage_level!r} is not a valid "
                f"thinking level; must be one of {sorted(_THINKING_LEVEL_NAMES)} or None"
            )


_validate_stage_thinking_levels()


def for_stage(
    llm: SupportsLLM,
    stage: Literal["extract", "bind", "quantity", "stated_words", "exclusion_read"],
) -> SupportsLLM:
    """Return ``llm`` or a shallow copy with the stage's thinking level.

    The original object wins when the stage has no override, the LLM is a
    test double, or its instance/config already sets a level. Otherwise the
    copy pins the stage value without rerunning ``__init__``'s config fallback.
    It shares the original client, connection pool, and concurrency cap.
    """
    level = _STAGE_THINKING_LEVELS.get(stage)
    if level is None or not isinstance(llm, LLM) or llm._thinking_level is not None:
        return llm
    clone = copy.copy(llm)
    clone._thinking_level = level
    return clone


def reseeded(llm: SupportsLLM) -> SupportsLLM:
    """`llm` sampling under another seed: a shallow copy whose seed is one past `llm`'s, or
    1 for an unseeded client. It shares the original's client, memo and extraction store,
    and the seed enters the memo key (`_memo_key`), so the memo keeps the two answers apart.
    Any other `SupportsLLM` comes back as it is."""
    if not isinstance(llm, LLM):
        return llm
    clone = copy.copy(llm)
    clone._seed = 1 if llm._seed is None else llm._seed + 1
    return clone


class SecondSampleCut(Exception):
    """A second sample still running once its first was read and it reached the earlier of
    its deadlines: `llm_second_sample_settle_s` after that read, widened by a memo-served
    first's live time, or `llm_timeout_s` from its own start (`Agreement.read_first`).
    Its read is not judged."""


# How a read's two samples ended (`Ended`): judged `agreed` or `disagreed`, or unjudged, the
# second `cut` (past the settle bound or its model deadline once its first was read, or
# cancelled with its request) or `failed` (a model error or refusal, an extraction-store
# write that failed, or a defect).
type Outcome = Literal["agreed", "disagreed", "cut", "failed"]


@dataclass(frozen=True, slots=True)
class Ended:
    """How a read's second sample ended, recorded once by its own task (`Agreement`):
    the outcome, what the second sample cost, and the exception a defect raised, set only
    with outcome `failed`."""

    outcome: Outcome
    usage: dict | None
    defect: Exception | None = None


# The end of a second sample cancelled or cut: its request was sent, and no response reports
# its tokens.
_CUT = Ended("cut", {UNCOUNTED_CALLS: 1})


class Agreement:
    """The second sample of a read (`two_samples`) and how it ended (`ended`).

    Its task runs the call under a deadline, `llm_timeout_s` from its start or earlier, the
    cut `read_first` sets once the first sample is read; either reschedules the task's own
    `asyncio.timeout`, so the cut stops the call wherever it is. Once the call answers and
    the first is read, the task judges the second against the first (`read_first`'s judge,
    which writes an agreed read to the extraction store). It returns how the read ended
    (`Ended`) and raises nothing but its cancellation, so each ending is classified, judged
    and logged once, by the task:

    - Once the first is read, either deadline cuts the sample (`SecondSampleCut`) and logs
      one info line naming `stage`, the deadline and the time it allowed. Before then the
      model's own deadline is a model failure (`LLMInfraError`), as a memo follower's is.
    - A model failure or refusal (`LLMInfraError`, `LLMOutputRefused`) is `failed` and logs
      a warning naming `stage`.
    - A judge whose extraction-store write fails (`OSError`) is `failed` and logs an error
      naming `stage`.
    - Any other exception, from the call or the judge, is a defect: `failed` with the
      exception kept (`Ended.defect`), logged once at error with its traceback and `stage`.
      Inside a request the reading is unjudged and the answer intact; a caller outside one
      has it raised (`two_samples`)."""

    def __init__(
        self, call: Callable[[], Coroutine[Any, Any, Sampled[Any]]], *, stage: str
    ) -> None:
        self.stage = stage
        self._model_at: float | None = None
        self._cut_at: float | None = None
        # When the first sample was read (`read_first`).
        self._read_at: float | None = None
        self._scope: asyncio.Timeout | None = None
        # The judge of the second's answer, set with the cut when the first is read.
        self._judge: asyncio.Future[Callable[[Any], bool]] = (
            asyncio.get_running_loop().create_future()
        )
        self.task: asyncio.Task[Ended] = asyncio.create_task(self._run(call))

    def _when(self) -> float | None:
        ends = [at for at in (self._model_at, self._cut_at) if at is not None]
        return min(ends, default=None)

    def _clamped(self) -> None:
        """Keep the cut no later than the model's own deadline."""
        if self._cut_at is not None and self._model_at is not None:
            self._cut_at = min(self._cut_at, self._model_at)

    def read_first(self, judge: Callable[[Any], bool], *, delay_s: float) -> None:
        """Take the first sample's `judge` of the second's answer and end the sample
        `delay_s` from now, or at the model's own deadline if that comes first. A sample
        already ending, or ended, keeps its end."""
        self._read_at = asyncio.get_running_loop().time()
        self._cut_at = self._read_at + delay_s
        self._clamped()
        if self._scope is not None and not self._scope.expired():
            self._scope.reschedule(self._when())
        self._judge.set_result(judge)

    async def ended(self) -> Ended:
        """How the sample ended, awaiting its task: `cut` for a cancelled one. Every call
        returns the same value. Cancelling this await leaves the task running for its
        holder to close (`SecondSamples.aclose`)."""
        if not self.task.done():
            await asyncio.wait([self.task])
        return _ended(self.task)

    async def _run(self, call: Callable[[], Coroutine[Any, Any, Sampled[Any]]]) -> Ended:
        try:
            second = await self._sample(call)
        except SecondSampleCut:
            return _CUT
        except (LLMInfraError, LLMOutputRefused) as exc:
            logger.warning(
                "%s: second sample failed; the read is unjudged",
                self.stage,
                exc_info=exc,
                extra={JSON_FIELDS_ATTR: {"stage": self.stage}},
            )
            refused = exc.usage if isinstance(exc, LLMOutputRefused) else None
            return Ended("failed", refused or {UNCOUNTED_CALLS: 1})
        except Exception as defect:
            return self._defect(defect, usage={UNCOUNTED_CALLS: 1})
        judge = await self._judge
        try:
            agreed = judge(second.answer)
        except OSError:
            logger.exception(
                "%s: storing the agreed read failed; the read is unjudged",
                self.stage,
                extra={JSON_FIELDS_ATTR: {"stage": self.stage}},
            )
            return Ended("failed", second.usage)
        except Exception as defect:
            return self._defect(defect, usage=second.usage)
        return Ended("agreed" if agreed else "disagreed", second.usage)

    def _defect(self, defect: Exception, *, usage: dict | None) -> Ended:
        logger.error(
            "%s: second sample raised a defect; the read is unjudged",
            self.stage,
            exc_info=defect,
            extra={JSON_FIELDS_ATTR: {"stage": self.stage}},
        )
        return Ended("failed", usage, defect=defect)

    async def _sample(self, call: Callable[[], Coroutine[Any, Any, Sampled[Any]]]) -> Sampled[Any]:
        """`call()` within this deadline. Cancelling a memo leader's caller releases its
        lease, and the memo then cancels the live call (`_lead`). The coroutine is made
        here, so a task cancelled before it starts leaves none unawaited."""
        self._model_at = asyncio.get_running_loop().time() + config.settings().llm_timeout_s
        self._clamped()
        try:
            async with asyncio.timeout_at(self._when()) as scope:
                self._scope = scope
                try:
                    return await call()
                finally:
                    self._scope = None
        except TimeoutError as exc:
            if self._cut_at is not None and self._read_at is not None:
                # The first was read, so the read goes unjudged whichever deadline ended it.
                # A first read past the model's deadline allows the second no time.
                deadline = "llm_timeout_s" if self._cut_at >= self._model_at else "settle_bound"
                allowed_s = max(0.0, self._cut_at - self._read_at)
                logger.info(
                    "%s: second sample reached its %s %.2f s after its first was read; cut, "
                    "its read unjudged",
                    self.stage,
                    "llm_timeout_s" if deadline == "llm_timeout_s" else "settle bound",
                    allowed_s,
                    extra={
                        JSON_FIELDS_ATTR: {
                            "stage": self.stage,
                            "deadline": deadline,
                            "allowed_s": allowed_s,
                        }
                    },
                )
                raise SecondSampleCut from exc
            raise LLMInfraError(
                f"Gemini API error: {_LLM_WAITER_DEADLINE_EXCEEDED_DETAIL}"
            ) from exc


def _ended(task: asyncio.Task[Ended]) -> Ended:
    """How a finished second sample's task ended (`Agreement`): `cut` when cancelled."""
    return _CUT if task.cancelled() else task.result()


class SecondSamples:
    """The second samples one request's reads start (`two_samples`), owned until the
    request ends. `bind` gives a copy of an `LLM` whose reads hand their second sample to
    this holder and serve their first at once; `aclose` cancels every second sample still
    running and awaits it, so no model call outlives the request, and a sample started
    after it is cancelled before it calls the model. `usage` sums what the second samples of
    served first samples cost. `CacheScope` holds one per request and closes it on every
    exit (`CacheScope.release`).

    A second sample runs at most `llm_second_sample_settle_s` after its first is read
    (`Agreement.read_first`); past it the sample is cut, holding its pool connection no
    longer, and its read is not judged. A second that answers in time is judged at once,
    and an agreed read is written to the extraction store then, whatever happens to the
    request after. An SSE answer never waits for a second sample. `done`, and so the
    buffered JSON response, waits at most that bound after the extraction's first sample,
    for layer 1 stores the answer only once every agreement has ended. A first sample the
    LLM memo serves returns at once, so its second also gets the time the memo's live call
    took (`two_samples`): a buffered repeat of a wording whose second was cut waits about
    one model call once, its second then reaches the memo, and later repeats hit layer 1. A
    second that takes longer than that and the bound is cut on every repeat, which is what
    the bound is for."""

    def __init__(self) -> None:
        self._started: list[Agreement] = []
        self._served: list[Agreement] = []
        self._closed = False

    def bind(self, llm: SupportsLLM) -> SupportsLLM:
        """`llm` holding its reads' second samples here: a shallow copy sharing its client,
        memo and extraction store. Any other `SupportsLLM` comes back as it is."""
        if not isinstance(llm, LLM):
            return llm
        clone = copy.copy(llm)
        clone._second_samples = self
        return clone

    def start(
        self, call: Callable[[], Coroutine[Any, Any, Sampled[Any]]], *, stage: str
    ) -> Agreement:
        """Run `call()` as a second sample held here (`Agreement`), cancelled before it
        calls the model when this holder is closed."""
        agreement = Agreement(call, stage=stage)
        if self._closed:
            agreement.task.cancel()
        self._started.append(agreement)
        return agreement

    def count(self, agreement: Agreement) -> None:
        """Count `agreement`'s second sample in `usage`: its first was served. A second
        whose first failed is counted with that failure by the caller."""
        self._served.append(agreement)

    def usage(self) -> dict | None:
        """The summed usage of the counted second samples that have finished
        (`Ended.usage`); after `aclose`, all of them."""
        total: dict | None = None
        for agreement in self._served:
            if agreement.task.done():
                total = sum_usage(total, _ended(agreement.task).usage)
        return total

    async def aclose(self) -> None:
        """Close the holder, cancel every second sample still running and await them."""
        self._closed = True
        running = [agreement.task for agreement in self._started if not agreement.task.done()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)

    def adopt(self, other: SecondSamples) -> None:
        """Take over `other`'s second samples, the counted ones included, so this holder's
        `aclose` and `usage` cover them and `other` holds none. A closed holder cancels the
        ones still running."""
        self._started.extend(other._started)
        self._served.extend(other._served)
        other._started, other._served = [], []
        if self._closed:
            for agreement in self._started:
                agreement.task.cancel()


def held_by(llm: SupportsLLM) -> SecondSamples | None:
    """The request holder `llm` hands its reads' second samples to (`SecondSamples.bind`),
    None when it settles each read before returning."""
    return llm._second_samples if isinstance(llm, LLM) else None


@dataclass(frozen=True)
class FixedAgreement:
    """A verdict known without a second sample: a stored read, or one built outside
    `two_samples`."""

    verdict: bool

    async def ended(self) -> Ended:
        return Ended("agreed" if self.verdict else "disagreed", None)


# The agreement of a read's samples, judged or known (`settle`).
type AnyAgreement = Agreement | FixedAgreement


async def settle(agreements: Sequence[AnyAgreement]) -> bool | None:
    """The joint verdict of `agreements` (`Agreement.ended`), each already bounded by its
    second's own deadline: False when any read disagreed, True when every read agreed, and
    None otherwise, some read being unjudged. Each ending logs where it happens."""
    outcomes = [(await agreement.ended()).outcome for agreement in agreements]
    if "disagreed" in outcomes:
        return False
    return True if all(outcome == "agreed" for outcome in outcomes) else None


async def two_samples[T, R](
    sample: Callable[[SupportsLLM], Coroutine[Any, Any, Sampled[T]]],
    llm: SupportsLLM,
    *,
    stage: str,
    read: Callable[[T], R],
    judge: Callable[[T, R, T], bool],
) -> tuple[R, dict | None, Agreement]:
    """Two concurrent samples of one call, `sample(llm)` and `sample(reseeded(llm))`: the
    first as `read` makes it, its usage, and the `Agreement` that judges the second against
    it (`judge(first, read(first), second)`). A first sample that fails, or that `read`
    refuses, raises and cancels the second, which is then never judged. Once the first is
    read, the second runs at most `llm_second_sample_settle_s` more, and at most
    `llm_timeout_s` from its start (`Agreement.read_first`). A first served from the memo
    took no model call, so the second also gets what the call the memo kept took live beyond
    the first's wait (`Sampled.memo_live_s`): a fresh call of the same wording needs about
    that long. A second slower than that and the bound is cut on every repeat.

    An `llm` bound to a request's `SecondSamples` returns once the first is read, with the
    first's usage: the second runs on under that holder, which counts its cost and closes
    it when the request ends, and is judged as soon as it answers. Any other `llm` awaits
    the second's end before returning, and the usage adds the second's (`Ended.usage`); a
    defect in the second sample or the judge, logged where it happened, raises here.
    `stage` names the call in the lines a cut, failed or defective second logs."""
    held = held_by(llm)
    samples = held if held is not None else SecondSamples()
    try:
        agreement = samples.start(lambda: sample(reseeded(llm)), stage=stage)
        started = asyncio.get_running_loop().time()
        try:
            first = await sample(llm)
            served = read(first.answer)
        except BaseException:
            agreement.task.cancel()
            await asyncio.gather(agreement.task, return_exceptions=True)
            raise
        waited_s = asyncio.get_running_loop().time() - started
        agreement.read_first(
            lambda second: judge(first.answer, served, second),
            delay_s=max(0.0, (first.memo_live_s or 0.0) - waited_s)
            + config.settings().llm_second_sample_settle_s,
        )
        samples.count(agreement)
        if held is not None:
            return served, first.usage, agreement
        ended = await agreement.ended()
        if ended.defect is not None:
            raise ended.defect
        return served, sum_usage(first.usage, ended.usage), agreement
    finally:
        if held is None:
            await samples.aclose()


def _thinking_config_for_model(
    model: str, level_override: str | None = None
) -> ThinkingConfig | None:
    """Build a model-specific ``ThinkingConfig``, or return ``None``.

    Gemini 3.5+ defaults to ``LOW``. Older Gemini 3 and unknown Gemini models
    default to ``MINIMAL``. Latest aliases use the same generation check.
    Gemini 2.5 uses ``thinking_budget=0``. Gemma and non-Gemini models omit
    thinking config. A level override applies to branches that set
    ``thinking_level`` because budget and level are mutually exclusive.
    """
    from google.genai.types import ThinkingConfig, ThinkingLevel

    def _level(default: ThinkingLevel) -> ThinkingLevel:
        if level_override is None:
            return default
        return getattr(ThinkingLevel, _THINKING_LEVEL_NAMES[level_override])

    if _is_gemini_3_5_plus(model):
        return ThinkingConfig(thinking_level=_level(ThinkingLevel.LOW))
    if "gemini-3" in model:
        return ThinkingConfig(thinking_level=_level(ThinkingLevel.MINIMAL))
    if "gemini-2.5" in model:
        return ThinkingConfig(thinking_budget=0, include_thoughts=False)
    if "gemma" in model:
        return None
    if "gemini" in model:
        # Unknown gemini-* models use a forward-compatible default.
        return ThinkingConfig(thinking_level=_level(ThinkingLevel.MINIMAL))
    return None


SCHEMA_REFUSAL = "schema_refusal"
"""The pydantic error type of a rule a response schema refuses an answer on, raised as
`pydantic_core.PydanticCustomError(SCHEMA_REFUSAL, ...)`. Such an answer is the model's,
so it raises `LLMOutputRefused`, which a caller may answer for and the response cache
stores; any other validation error is an infrastructure failure."""


def _parse[SchemaT: BaseModel](
    text: str | bytes | None, schema: type[SchemaT], *, model: str, usage: dict | None
) -> SchemaT:
    """Validate the model's `text` against `schema`, the one parse every holder of a call
    runs: the live caller, a memo leader, its followers and a memo hit.

    Raises `LLMOutputRefused` with `usage` when every validation error is of type
    `SCHEMA_REFUSAL`. Empty or unparsable text, a missing field, a wrong type, or a value
    past a length limit raises `LLMInfraError`."""
    detail = f"LLM output failed schema validation (model={model!r}, schema={schema.__name__!r})"
    if not text:
        raise LLMInfraError(detail)
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        if all(error["type"] == SCHEMA_REFUSAL for error in exc.errors()):
            raise LLMOutputRefused(detail, usage=usage) from exc
        raise LLMInfraError(detail) from exc


class SupportsLLM(Protocol):
    """Structural interface for injected LLM clients and test doubles."""

    async def generate_structured(
        self,
        *,
        prompt: str,
        system: str,
        schema: type[_SchemaT],
    ) -> tuple[_SchemaT, dict | None]: ...


class SpentUsage:
    """The token usage of each model call made inside `recording_llm_usage`."""

    def __init__(self) -> None:
        self.calls: list[dict] = []


_SPENT: ContextVar[SpentUsage | None] = ContextVar("llm_spent_usage", default=None)


@contextmanager
def recording_llm_usage() -> Iterator[SpentUsage]:
    """Collect, for the body of the `with` block and the tasks it starts, the usage of every
    call `generate_structured_async` makes, a refused answer's included. An enclosing record
    gains them when the block closes."""
    outer = _SPENT.get()
    record = SpentUsage()
    token = _SPENT.set(record)
    try:
        yield record
    finally:
        _SPENT.reset(token)
        if outer is not None:
            outer.calls.extend(record.calls)


@contextmanager
def unrecorded_llm_usage() -> Iterator[None]:
    """Keep the usage of every call in the body of the `with` block out of the enclosing
    `recording_llm_usage` record: the calls of a detached comparison are not the answer's."""
    token = _SPENT.set(None)
    try:
        yield
    finally:
        _SPENT.reset(token)


def detached(llm: SupportsLLM) -> SupportsLLM:
    """`llm` for a call no answer waits on: a shallow copy with no memo, so the call reaches
    the model even when an identical request is in flight or stored, on the detached
    client, whose connections are its own, so the call never takes a served call's
    connection. Any other `SupportsLLM` comes back as it is."""
    if not isinstance(llm, LLM):
        return llm
    clone = copy.copy(llm)
    clone._memo = None
    clone._detached = True
    return clone


def _note_spent(usage: dict | None) -> None:
    record = _SPENT.get()
    if record is not None and usage is not None:
        record.calls.append(usage)


class Sampled[T](NamedTuple):
    """One structured-output answer (`generate_structured_async`): the answer, its usage, and, for
    an answer the LLM memo served, how long the live call it kept took (`_memo_value`),
    None for a call made live."""

    answer: T
    usage: dict | None
    memo_live_s: float | None


async def generate_structured_async[SchemaT: BaseModel](
    llm: SupportsLLM,
    *,
    prompt: str,
    system: str,
    schema: type[SchemaT],
) -> Sampled[SchemaT]:
    """Run one structured-output request through `_generate_structured`, and note its usage
    in the enclosing `recording_llm_usage` record."""
    try:
        sampled = await _generate_structured(llm, prompt=prompt, system=system, schema=schema)
    except LLMOutputRefused as refusal:
        _note_spent(refusal.usage)
        raise
    _note_spent(sampled.usage)
    return sampled


async def _generate_structured[SchemaT: BaseModel](
    llm: SupportsLLM,
    *,
    prompt: str,
    system: str,
    schema: type[SchemaT],
) -> Sampled[SchemaT]:
    """Run one structured-output request, single-flighted when memoised.

    The resolved request forms the memo key. The memo holds the model's text with how long
    the live call took (`_memo_value`). A hit returns freshly validated JSON with zero
    usage and that live time. One leader calls Gemini in a shielded task and publishes the
    model's text, an answer its schema refuses included. Followers wait up to
    ``llm_timeout_s`` and parse that text as the leader did, with its live time, so a
    refusal reaches them as ``LLMOutputRefused``. Other leader failures reach followers as
    ``LLMInfraError`` without another Gemini call.

    Every holder releases its lease on success, failure, timeout, or
    cancellation. The leader serves followers until its last lease is released.
    The memo then detaches the key and cancels the call. LLMs without a memo,
    including test doubles, run directly and cancellation stops their request.
    """
    if not isinstance(llm, LLM) or llm._memo is None:
        answer, usage = await llm.generate_structured(prompt=prompt, system=system, schema=schema)
        return Sampled(answer, usage, None)

    memo = llm._memo
    config_obj = llm._resolve_config(system=system, schema=schema)
    key = _memo_key(model=llm._model, prompt=prompt, schema=schema, config=config_obj)

    stored, lease = await memo.acquire(key)
    if stored is not None:
        logger.info("LLM memo hit: schema=%s", schema.__name__)
        # Each hit builds a fresh instance so callers can mutate their results independently.
        return _served(stored, schema, model=llm._model)

    assert lease is not None  # acquire() returns a lease exactly when it missed
    if lease.is_leader:
        answer, usage = await _lead(
            memo, lease, llm=llm, prompt=prompt, schema=schema, config=config_obj
        )
        return Sampled(answer, usage, None)

    try:
        published = await asyncio.wait_for(
            memo.await_leader(lease), timeout=config.settings().llm_timeout_s
        )
    except TimeoutError as exc:
        # await_leader raises TimeoutError at the follower's deadline. Live Gemini
        # timeouts use the same type, giving followers and leaders one outcome.
        raise LLMInfraError(f"Gemini API error: {_LLM_WAITER_DEADLINE_EXCEEDED_DETAIL}") from exc
    finally:
        memo.leave(lease)
    if published is None:
        # A failed leader stores nothing and hands every parked caller None.
        # Re-acquiring the key here would re-issue the Gemini call that is
        # failing, once per parked request.
        raise LLMInfraError(f"Gemini API error: {_LLM_MEMO_LEADER_FAILED_DETAIL}")
    # Followers report zero usage. The leader's live call reports its token cost.
    logger.info("LLM memo hit: schema=%s", schema.__name__)
    return _served(published.value, schema, model=llm._model)


# A memo value: the live call's latency in seconds, a little-endian double, then the model's
# text (`_memo_value`).
_LIVE_S = struct.Struct("<d")


def _memo_value(text: str, *, live_s: float) -> bytes:
    """The memo's value for a live call that answered `text` `live_s` seconds after it
    left the connection pool's queue (`_CallTiming`)."""
    return _LIVE_S.pack(live_s) + text.encode()


def _served[SchemaT: BaseModel](
    value: bytes, schema: type[SchemaT], *, model: str
) -> Sampled[SchemaT]:
    """The answer a memo value holds (`_memo_value`), for a caller that made no call: its
    text parsed, zero usage, and the live call's time. A refusal reaches that caller as
    `LLMOutputRefused` with zero usage."""
    (live_s,) = _LIVE_S.unpack_from(value)
    answer = _parse(value[_LIVE_S.size :], schema, model=model, usage=dict(MEMO_HIT_USAGE))
    return Sampled(answer, dict(MEMO_HIT_USAGE), live_s)


# The loop keeps weak task refs. This set retains leaders serving followers after cancellation.
_INFLIGHT_LEADERS: set[asyncio.Task] = set()


async def _lead[SchemaT: BaseModel](
    memo: LLMMemo,
    lease: Lease,
    *,
    llm: LLM,
    prompt: str,
    schema: type[SchemaT],
    config: GenerateContentConfig,
) -> tuple[SchemaT, dict | None]:
    """Run the live call in a shielded task that publishes or abandons its key.

    Caller cancellation releases only its lease. The task serves remaining
    followers. The orphan callback cancels it when the last holder leaves.
    """

    async def publish(text: str, live_s: float) -> None:
        # Preserve the caller's outcome if memo publication fails.
        try:
            payload = _memo_value(text, live_s=live_s)
            await memo.publish(lease, payload, size=len(payload))
        except BaseException:
            logger.exception(
                "LLM memo publish failed after a successful fetch; serving "
                "this caller's answer uncached and freeing the key"
            )
            await memo.abandon(lease)

    async def lead_and_settle():
        # The memo keeps how long the model took (`_CallTiming`), which a second sample
        # served this answer's twin is given (`two_samples`).
        timing = _CallTiming()
        token = _CALL_TIMING.set(timing)
        started = asyncio.get_running_loop().time()
        try:
            text, usage = await llm._fetch_text(prompt=prompt, config=config)
        except BaseException:
            await memo.abandon(lease)
            raise
        finally:
            _CALL_TIMING.reset(token)
        live_s = asyncio.get_running_loop().time() - (timing.sent_at or started)
        try:
            parsed = _parse(text, schema, model=llm._model, usage=usage)
        except LLMOutputRefused:
            # A refusal is the model's answer, so the memo stores it like any other.
            await publish(text, live_s)
            raise
        except BaseException:
            await memo.abandon(lease)
            raise
        await publish(text, live_s)
        return parsed, usage

    task = asyncio.create_task(lead_and_settle())
    _INFLIGHT_LEADERS.add(task)
    task.add_done_callback(_forget_leader)

    def cancel_orphan() -> None:
        logger.info("LLM memo cancelled an orphaned Gemini call: schema=%s", schema.__name__)
        task.cancel()

    # Registered before the first await, so the callback is in place before
    # any caller holding this flight can leave it.
    memo.on_orphaned(lease, cancel_orphan)
    try:
        return await asyncio.shield(task)
    finally:
        memo.leave(lease)


def _forget_leader(task: asyncio.Task) -> None:
    """Drop the leader reference and consume an exception with no awaiter.

    This prevents asyncio's unretrieved-exception warning for a leader that
    outlives its caller. Cancelled orphan tasks need no exception retrieval.
    """
    _INFLIGHT_LEADERS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.info("LLM memo leader call failed with no awaiter left: %s", exc)


# The startup key check's whole models.get, connect to last byte. HttpOptions.timeout bounds
# each phase of a request alone, so the bound is set around the call.
_WARM_TIMEOUT_S = 10.0

# The client errors that reject the key or the model outright: 400 INVALID_ARGUMENT for an
# invalid key, 401 and 403 for credentials Gemini refuses, 404 for an unknown model.
_WARM_REJECTION_CODES = frozenset({400, 401, 403, 404})


async def warm() -> None:
    """Build the singleton at startup and check the key with one models.get on the engine
    model, which spends no tokens.

    A missing key or dependency, and a client error in ``_WARM_REJECTION_CODES``, raise
    ``LLMInfraError``, so the instance never serves. Any other failure, a 5xx, a 408 or 429,
    a transport error, an answer the SDK cannot read or ``_WARM_TIMEOUT_S`` passing, logs a
    WARNING and returns: the instance serves, and searches answer 503 until Gemini answers.
    """
    client = _get_client()
    from google.genai import errors

    model = config.settings().engine_model
    try:
        async with asyncio.timeout(_WARM_TIMEOUT_S):
            await client.aio.models.get(model=model)
    except Exception as exc:
        if isinstance(exc, errors.ClientError) and exc.code in _WARM_REJECTION_CODES:
            raise LLMInfraError(f"Gemini rejected the startup key check: {exc}") from exc
        # Searches report any other failure, request by request, as they did before the check.
        logger.warning("Gemini key check failed; serving without it: %r", exc)
    else:
        logger.info("Gemini key check passed (model=%s)", model)


# LLMMemo stores, by `_memo_key`, an envelope of the model's text and how long its live call
# took (`_memo_value`). ResponseCache supplies the byte limits, TTL, and single-flight
# behavior.
LLMMemo = ResponseCache


def build_memo() -> LLMMemo | None:
    """Build a configured memo, or return ``None`` when max bytes is ``0``.

    Construct ``LLMMemo`` directly to set custom bounds.
    """
    max_bytes = config.settings().llm_memo_max_bytes
    if max_bytes == 0:
        return None
    return LLMMemo(
        max_bytes=max_bytes,
        max_entry_bytes=config.settings().llm_memo_max_entry_bytes,
        ttl_s=config.settings().llm_memo_ttl_s,
    )


@functools.cache
def schema_digest(schema: type[BaseModel]) -> str:
    """SHA-256 of `schema`'s JSON schema, computed once per class."""
    canonical = json.dumps(schema.model_json_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _memo_key(
    *,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    config: GenerateContentConfig,
) -> str:
    """Hash the resolved model, prompt, schema, and generation config.

    The config dump excludes ``response_schema``. The schema enters as its
    qualified name and a hash of its JSON schema, so a field added without a
    rename moves the key. Other config fields, including the system
    instruction, temperature, thinking config, and seed, enter the key.
    A floating model alias keeps its key until the memo entry expires.

    ASCII-escaped JSON keys any prompt, a lone surrogate included, though the
    request models refuse one in every text a prompt is built from.
    """
    document = {
        "model": model,
        "prompt": prompt,
        "schema": f"{schema.__module__}.{schema.__qualname__}",
        "schema_fields": schema_digest(schema),
        "config": config.model_dump(mode="json", exclude={"response_schema"}),
    }
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "llm_" + hashlib.sha256(canonical).hexdigest()


class _ApiKeyClients:
    """The two clients of an `LLM` built with its own api key, the served calls' and the
    detached calls', each built on first use with the pool size its singleton has. Clones of
    the `LLM` hold this same object, so they share both clients."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @functools.cached_property
    def served(self) -> genai.Client:
        return _build_client(self._api_key, max_connections=config.settings().llm_max_concurrency)

    @functools.cached_property
    def detached(self) -> genai.Client:
        return _build_client(
            self._api_key, max_connections=config.settings().jev_shadow_max_inflight
        )


class LLM:
    """Async Gemini structured-output wrapper over ``client.aio.models``.

    Usage::

        llm = LLM()
        result: MySchema = await llm.generate_structured(
            prompt="...", system="...", schema=MySchema
        )

    Calls are always live. The engine uses ``generate_structured_async`` for
    memoisation. ``extraction_store`` holds the extractions ``extract()`` reads
    before it calls the model. Tests inject ``FakeLLM``. Subclasses that skip
    ``__init__`` inherit ``_memo = None`` and ``extraction_store = None`` and keep
    the live-call behaviour.
    """

    # Class-level defaults let subclasses that skip __init__ access both stores.
    # Store-carrying instances override them in __init__.
    _memo: LLMMemo | None = None
    extraction_store: ExtractionStore | None = None
    # The request's holder of second samples (`SecondSamples.bind`); None settles each
    # read's agreement before it returns (`two_samples`).
    _second_samples: SecondSamples | None = None
    _detached = False

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
        thinking_level: str | None = None,
        memo: LLMMemo | None = None,
        extraction_store: ExtractionStore | None = None,
    ):
        # An instance api_key gets its own clients, each built on its first call and shared
        # with the instance's clones. Otherwise _client() uses the singletons.
        self._model = model or config.settings().engine_model
        self._own_clients = _ApiKeyClients(api_key) if api_key else None
        # Explicit seed and thinking_level kwargs override settings.
        # Unset kwargs use process defaults.
        self._seed = seed if seed is not None else config.settings().llm_seed
        self._thinking_level = (
            thinking_level if thinking_level is not None else config.settings().llm_thinking_level
        )
        # A shared memo serves both app surfaces, and for_stage clones share it.
        # None keeps every call live.
        self._memo = memo
        self.extraction_store = extraction_store

    def sampling_identity(self) -> dict[str, object]:
        """The model id, seed and thinking level that, with the request, decide the answer."""
        return {"model": self._model, "seed": self._seed, "thinking_level": self._thinking_level}

    def _client(self) -> genai.Client:
        own = self._own_clients
        if self._detached:
            return own.detached if own is not None else _get_detached_client()
        return own.served if own is not None else _get_client()

    def _resolve_config(self, *, system: str, schema: type[BaseModel]) -> GenerateContentConfig:
        """Build the config shared by live calls and the memo key."""
        try:
            from google.genai.types import GenerateContentConfig
        except ImportError as exc:
            raise LLMInfraError(
                "search.engine requires the 'engine' extra. "
                "Install with: uv sync --extra engine (or pip install 'search[engine]')."
            ) from exc

        config_kwargs: dict = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            # Pass the Pydantic class directly so response.parsed is a typed instance.
            "response_schema": schema,
            "temperature": _temperature_for_model(self._model),
            "thinking_config": _thinking_config_for_model(self._model, self._thinking_level),
        }
        # Seed applies to every model. It is omitted only when the kwarg and setting are unset.
        if self._seed is not None:
            config_kwargs["seed"] = self._seed
        return GenerateContentConfig(**config_kwargs)

    async def generate_structured(
        self,
        *,
        prompt: str,
        system: str,
        schema: type[_SchemaT],
    ) -> tuple[_SchemaT, dict | None]:
        """Call Gemini and return ``(parsed, usage)``.

        ``usage`` has ``input_tokens``, ``output_tokens``, and ``cached_tokens`` keys, or ``None``
        when Gemini omits usage metadata. This method always makes a live call.
        the engine uses ``generate_structured_async`` for memoisation.
        Cancellation stops the underlying HTTP request. Transport/API errors
        and text the schema cannot read raise ``LLMInfraError``; an answer the
        schema refuses on purpose raises ``LLMOutputRefused``.
        """
        text, usage = await self._fetch_text(
            prompt=prompt, config=self._resolve_config(system=system, schema=schema)
        )
        return _parse(text, schema, model=self._model, usage=usage), usage

    async def _fetch_text(
        self, *, prompt: str, config: GenerateContentConfig
    ) -> tuple[str, dict | None]:
        """One live generate_content call: the model's text, empty when it sent none, and
        its usage. `_parse` reads the text against the request's schema."""
        client = self._client()
        try:
            response = await client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            raise LLMInfraError(f"Gemini API error: {exc}") from exc

        usage: dict | None = None
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            usage = {
                "input_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
                "cached_tokens": getattr(usage_meta, "cached_content_token_count", 0) or 0,
            }
            total = usage["input_tokens"] + usage["output_tokens"]
            cache_hit_pct = (
                (usage["cached_tokens"] / usage["input_tokens"] * 100)
                if usage["input_tokens"] > 0
                else 0.0
            )
            logger.info(
                "LLM usage: input=%d output=%d cached=%d total=%d cache_hit_pct=%.1f%%",
                usage["input_tokens"],
                usage["output_tokens"],
                usage["cached_tokens"],
                total,
                cache_hit_pct,
            )

        return response.text or "", usage
