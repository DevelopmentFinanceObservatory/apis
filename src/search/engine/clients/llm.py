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
import hashlib
import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

import search.engine.config as config  # module-attribute access for runtime-resolved values
from search.engine.errors import LLMInfraError, LLMOutputRefused
from search.engine.response_cache import Lease, ResponseCache

if TYPE_CHECKING:
    from google import genai
    from google.genai.types import GenerateContentConfig, ThinkingConfig

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

logger = logging.getLogger(__name__)

# Lazy singleton, built on first call to generate_structured.
_CLIENT = None

# Memo hits report zero usage because they skip Gemini. Live calls return None
# when Gemini omits usage metadata.
_MEMO_HIT_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

# Follower wait limit. HttpOptions.timeout is per phase, so a stalled
# connection can outlast it. This overall wait uses the same configured bound.
_LLM_WAITER_DEADLINE_EXCEEDED_DETAIL = (
    "the Gemini structured-output call exceeded SEARCH_LLM_TIMEOUT_S"
)

# Followers receive this generic error when the leader fails.
# The leader's detail belongs to its own caller.
_LLM_MEMO_LEADER_FAILED_DETAIL = "the shared in-flight Gemini structured-output call failed"


class _LoopLocalTransport(httpx.AsyncBaseTransport):
    """Keep one ``max_connections`` pool per event loop.

    Connections belong to their creating loop, so calls on different loops
    need separate pools. Pools for closed loops are removed on the next lookup.
    The lock protects simultaneous loop access from different threads.
    HTTP/1.1 uses one request per connection, so the limit bounds each
    loop's concurrent Gemini requests. Pool waits use the request timeout.
    """

    def __init__(self, *, max_connections: int) -> None:
        self._lock = threading.Lock()
        self._limits = httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        )
        self._pools: dict[asyncio.AbstractEventLoop, httpx.AsyncHTTPTransport] = {}

    def _pool(self) -> httpx.AsyncHTTPTransport:
        loop = asyncio.get_running_loop()
        with self._lock:
            for closed in [known for known in self._pools if known.is_closed()]:
                del self._pools[closed]
            pool = self._pools.get(loop)
            if pool is None:
                pool = self._pools[loop] = httpx.AsyncHTTPTransport(limits=self._limits)
            return pool

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._pool().handle_async_request(request)

    async def aclose(self) -> None:
        with self._lock:
            pool = self._pools.pop(asyncio.get_running_loop(), None)
        if pool is not None:
            await pool.aclose()


def _build_client(api_key: str) -> genai.Client:
    """Build a Gemini client using one-attempt httpx requests.

    Passing an async httpx client selects google-genai's httpx path, avoiding
    aiohttp's 1-10 second retry. Passing the client itself also prevents
    google-genai from wrapping the shared transport in extra clients whose
    finalizers can close its loop's pool and exceed the connection cap.

    ``HttpOptions.timeout`` is converted from milliseconds to seconds and
    bounds connect, read, write, and pool waits. Saturated pools raise
    ``httpx.PoolTimeout``, translated to ``LLMInfraError`` by the caller.
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
        transport=_LoopLocalTransport(max_connections=config.settings().llm_max_concurrency),
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
    """Return (or build) the singleton genai.Client.

    Raises LLMInfraError only when a live call is attempted and GEMINI_API_KEY
    is absent. Tests use FakeLLM instead, bypassing this path.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = config.settings().gemini_api_key
        if not api_key:
            raise LLMInfraError(
                "GEMINI_API_KEY is required for live LLM calls. "
                "Set the environment variable or use FakeLLM in tests."
            )
        _CLIENT = _build_client(api_key)
    return _CLIENT


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

# Extract keeps the instance's resolved level. Bind and the quantity read default to minimal.
# An instance-level or config override takes precedence in for_stage().
_STAGE_THINKING_LEVELS: dict[str, str | None] = {
    "extract": None,
    "bind": "minimal",
    "quantity": "minimal",
}


def _validate_stage_thinking_levels() -> None:
    for stage_name, stage_level in _STAGE_THINKING_LEVELS.items():
        if stage_level is not None and stage_level not in _THINKING_LEVEL_NAMES:
            raise ValueError(
                f"_STAGE_THINKING_LEVELS[{stage_name!r}] = {stage_level!r} is not a valid "
                f"thinking level; must be one of {sorted(_THINKING_LEVEL_NAMES)} or None"
            )


_validate_stage_thinking_levels()


def for_stage(llm: SupportsLLM, stage: Literal["extract", "bind", "quantity"]) -> SupportsLLM:
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


async def generate_structured_async[SchemaT: BaseModel](
    llm: SupportsLLM,
    *,
    prompt: str,
    system: str,
    schema: type[SchemaT],
) -> tuple[SchemaT, dict | None]:
    """Run one structured-output request, single-flighted when memoised.

    The resolved request forms the memo key. A hit returns freshly validated
    JSON with zero usage. One leader calls Gemini in a shielded task and
    publishes the model's text, an answer its schema refuses included. Followers
    wait up to ``llm_timeout_s`` and parse that text as the leader did, so a
    refusal reaches them as ``LLMOutputRefused``. Other leader failures reach
    followers as ``LLMInfraError`` without another Gemini call.

    Every holder releases its lease on success, failure, timeout, or
    cancellation. The leader serves followers until its last lease is released.
    The memo then detaches the key and cancels the call. LLMs without a memo,
    including test doubles, run directly and cancellation stops their request.
    """
    if not isinstance(llm, LLM) or llm._memo is None:
        return await llm.generate_structured(prompt=prompt, system=system, schema=schema)

    memo = llm._memo
    config_obj = llm._resolve_config(system=system, schema=schema)
    key = _memo_key(model=llm._model, prompt=prompt, schema=schema, config=config_obj)

    stored, lease = await memo.acquire(key)
    if stored is not None:
        logger.info("LLM memo hit: schema=%s", schema.__name__)
        # Each hit builds a fresh instance so callers can mutate their results independently.
        return _served(stored, schema, model=llm._model), dict(_MEMO_HIT_USAGE)

    assert lease is not None  # acquire() returns a lease exactly when it missed
    if lease.is_leader:
        return await _lead(memo, lease, llm=llm, prompt=prompt, schema=schema, config=config_obj)

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
    value, _timed_out, _storable = published
    # Followers report zero usage. The leader's live call reports its token cost.
    logger.info("LLM memo hit: schema=%s", schema.__name__)
    return _served(value, schema, model=llm._model), dict(_MEMO_HIT_USAGE)


def _served[SchemaT: BaseModel](text: bytes, schema: type[SchemaT], *, model: str) -> SchemaT:
    """Parse the model's text the memo holds, for a caller that made no call. A refusal
    reaches that caller as `LLMOutputRefused` with zero usage."""
    return _parse(text, schema, model=model, usage=dict(_MEMO_HIT_USAGE))


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

    async def publish(text: str) -> None:
        # Preserve the caller's outcome if memo publication fails.
        try:
            payload = text.encode()
            await memo.publish(lease, payload, size=len(payload))
        except BaseException:
            logger.exception(
                "LLM memo publish failed after a successful fetch; serving "
                "this caller's answer uncached and freeing the key"
            )
            await memo.abandon(lease)

    async def lead_and_settle():
        try:
            text, usage = await llm._fetch_text(prompt=prompt, config=config)
        except BaseException:
            await memo.abandon(lease)
            raise
        try:
            parsed = _parse(text, schema, model=llm._model, usage=usage)
        except LLMOutputRefused:
            # A refusal is the model's answer, so the memo stores it like any other.
            await publish(text)
            raise
        except BaseException:
            await memo.abandon(lease)
            raise
        await publish(text)
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


def warm() -> None:
    """Build the singleton at startup. Missing keys or dependencies raise ``LLMInfraError``."""
    _get_client()


# LLMMemo stores the model's text as bytes by `_memo_key`. ResponseCache supplies the
# byte limits, TTL, and single-flight behavior.
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


def _memo_key(
    *,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    config: GenerateContentConfig,
) -> str:
    """Hash the resolved model, prompt, schema, and generation config.

    The config dump excludes ``response_schema``. The schema's qualified
    name identifies it within the process. Other config fields, including the
    system instruction, temperature, thinking config, and seed, enter the key.
    A floating model alias keeps its key until the memo entry expires.

    ASCII-escaped JSON handles lone surrogates that can survive request
    parsing. Such prompts form deterministic keys and reach Gemini as ordinary
    cache misses.
    """
    document = {
        "model": model,
        "prompt": prompt,
        "schema": f"{schema.__module__}.{schema.__qualname__}",
        "config": config.model_dump(mode="json", exclude={"response_schema"}),
    }
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "llm_" + hashlib.sha256(canonical).hexdigest()


class LLM:
    """Async Gemini structured-output wrapper over ``client.aio.models``.

    Usage::

        llm = LLM()
        result: MySchema = await llm.generate_structured(
            prompt="...", system="...", schema=MySchema
        )

    Calls are always live. The engine uses ``generate_structured_async`` for
    memoisation. Tests inject ``FakeLLM``. Subclasses that skip ``__init__``
    inherit ``_memo = None`` and keep the live-call behaviour.
    """

    # Class-level default lets subclasses that skip __init__ access `_memo`.
    # Memo-carrying instances override it in __init__.
    _memo: LLMMemo | None = None

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        seed: int | None = None,
        thinking_level: str | None = None,
        memo: LLMMemo | None = None,
    ):
        # An instance api_key creates one client shared with its stage clones. Otherwise _client()
        # uses the singleton.
        self._model = model or config.settings().engine_model
        self._override_client = _build_client(api_key) if api_key else None
        # Explicit seed and thinking_level kwargs override settings.
        # Unset kwargs use process defaults.
        self._seed = seed if seed is not None else config.settings().llm_seed
        self._thinking_level = (
            thinking_level if thinking_level is not None else config.settings().llm_thinking_level
        )
        # A shared memo serves both app surfaces, and for_stage clones share it.
        # None keeps every call live.
        self._memo = memo

    def _client(self) -> genai.Client:
        if self._override_client is not None:
            return self._override_client
        return _get_client()

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
