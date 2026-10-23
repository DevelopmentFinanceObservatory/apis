"""Tests memo keys, hits, byte limits, and single-flight cancellation with an offline client double.

Async calls use `asyncio.run` because the suite has no pytest-asyncio plugin.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai.types import GenerateContentConfig, ThinkingConfig, ThinkingLevel
from pydantic import BaseModel, create_model

import search.engine.clients.llm as llm_module
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.llm import (
    LLM,
    LLMMemo,
    _memo_key,
    build_memo,
    for_stage,
    generate_structured_async,
    recording_llm_usage,
)
from search.engine.errors import LLMInfraError, LLMOutputRefused
from search.engine.extract import Extraction
from tests.fixtures import FakeLLM
from tests.fixtures.keys import fixture_key

# How many bytes one `_DummySchema(value=1)` answer serialises to, so the
# byte-budget tests below can state an exact ceiling.
_ANSWER_BYTES = len('{"value":1}')


class _DummySchema(BaseModel):
    value: int = 0


class _OtherSchema(BaseModel):
    other: int = 0


class _FakeUsageMeta:
    prompt_token_count = 11
    candidates_token_count = 22
    cached_content_token_count = 0


class _FakeResponse:
    """The slice of a genai response `LLM._fetch_text` reads."""

    def __init__(self, *, text, usage_metadata=None) -> None:
        self.text = text
        self.model_version = "gemini-3.5-flash-lite-007"
        self.usage_metadata = usage_metadata


class _CountingModels:
    """Records request config and returns or raises `answer`."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str, object]] = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


class _CountingClient:
    def __init__(self, answer) -> None:
        self.models = _CountingModels(answer)
        # Both client paths share the recorder used by call-count assertions.
        self.aio = SimpleNamespace(models=self.models)


def _llm(monkeypatch, answer, **kwargs) -> tuple[LLM, _CountingClient]:
    client = _CountingClient(answer)
    monkeypatch.setattr(llm_module, "_get_client", lambda: client)
    return LLM(**kwargs), client


def _memo(**kwargs) -> LLMMemo:
    """A memo with generous defaults, overridable per test."""
    return LLMMemo(
        max_bytes=kwargs.pop("max_bytes", 1024 * 1024),
        max_entry_bytes=kwargs.pop("max_entry_bytes", 256 * 1024),
        ttl_s=kwargs.pop("ttl_s", 3600.0),
        **kwargs,
    )


def _call(llm, *, prompt="p", system="s", schema=_DummySchema):
    return generate_structured_async(llm, prompt=prompt, system=system, schema=schema)


def _response() -> _FakeResponse:
    return _FakeResponse(text='{"value":1}', usage_metadata=_FakeUsageMeta())


_LIVE_USAGE = {"input_tokens": 11, "output_tokens": 22, "cached_tokens": 0}
_HIT_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


def _fixture(schema_name: str) -> dict:
    """Load a recorded response for the named schema."""
    with open(Path(__file__).resolve().parents[1] / "fixtures" / "llm_responses.json") as f:
        responses = json.load(f)
    key = next(k for k in responses if k.startswith(f"{schema_name}@"))
    return responses[key]


# ---------------------------------------------------------------------------
# hit vs miss
# ---------------------------------------------------------------------------


def test_first_call_misses_and_second_identical_call_hits(monkeypatch):
    llm, client = _llm(monkeypatch, _response(), memo=_memo())

    first, first_usage = asyncio.run(_call(llm))
    second, second_usage = asyncio.run(_call(llm))

    assert len(client.models.calls) == 1
    assert first == second
    assert first_usage == _LIVE_USAGE
    assert second_usage == _HIT_USAGE


def test_memo_none_calls_every_time(monkeypatch):
    llm, client = _llm(monkeypatch, _response())

    asyncio.run(_call(llm))
    asyncio.run(_call(llm))

    assert len(client.models.calls) == 2


def test_disabled_memo_calls_every_time(monkeypatch, search_settings):
    """SEARCH_LLM_MEMO_MAX_BYTES=0 disables the memo: build_memo() returns
    None, so the LLM carries none and every call reaches Gemini."""
    search_settings(llm_memo_max_bytes=0)
    assert build_memo() is None

    llm, client = _llm(monkeypatch, _response(), memo=build_memo())

    asyncio.run(_call(llm))
    asyncio.run(_call(llm))

    assert len(client.models.calls) == 2


def test_a_memo_less_llm_and_fakellm_both_call_generate_structured_directly(monkeypatch):
    """FakeLLM and `LLM(memo=None)` bypass memo lookup and call the model each time."""
    llm, client = _llm(monkeypatch, _response())
    fake = FakeLLM(responses={fixture_key(_DummySchema, "s", "p"): {"value": 7}})

    assert asyncio.run(_call(llm))[0] == _DummySchema(value=1)
    assert asyncio.run(_call(fake))[0] == _DummySchema(value=7)  # ty: ignore[invalid-argument-type]  # structural double
    assert asyncio.run(_call(llm))[0] == _DummySchema(value=1)

    assert len(client.models.calls) == 2  # no memo, so both calls are live


def test_cancelling_a_memo_less_call_cancels_the_underlying_fetch(monkeypatch):
    """The no-memo path awaits the client coroutine, so cancellation reaches HTTP.
    A `to_thread` wrapper could only abandon it."""
    reached = asyncio.Event()
    cancelled_inside = asyncio.Event()

    class _HangingModels:
        async def generate_content(self, *, model, contents, config):
            reached.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled_inside.set()
                raise

    class _HangingClient:
        def __init__(self) -> None:
            self.aio = SimpleNamespace(models=_HangingModels())

    monkeypatch.setattr(llm_module, "_get_client", lambda: _HangingClient())

    async def scenario():
        llm = LLM()
        task = asyncio.create_task(_call(llm))
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task

    task = asyncio.run(scenario())

    assert task.cancelled()
    assert cancelled_inside.is_set()


def test_an_llm_subclass_that_skips_init_is_memo_free(monkeypatch):
    """Subclasses that skip `LLM.__init__` inherit `_memo=None` and bypass memoisation."""

    class _Wrapper(LLM):
        """Stands in for a subclass that sets its own attributes and never
        calls LLM.__init__."""

        def __init__(self) -> None:
            self._model = "gemini-3.5-flash-lite"

        async def generate_structured(self, *, prompt, system, schema):
            return schema(value=3), None

    assert "_memo" not in _Wrapper().__dict__
    assert asyncio.run(_call(_Wrapper()))[0] == _DummySchema(value=3)


def test_generate_structured_never_consults_a_memo(monkeypatch):
    """`generate_structured` always reaches the model, even when the instance has a memo."""
    memo = _memo()
    llm, client = _llm(monkeypatch, _response(), memo=memo)

    async def _twice():
        first = await llm.generate_structured(prompt="p", system="s", schema=_DummySchema)
        second = await llm.generate_structured(prompt="p", system="s", schema=_DummySchema)
        return first, second

    (first, first_usage), (second, second_usage) = asyncio.run(_twice())

    assert len(client.models.calls) == 2
    assert first == second == _DummySchema(value=1)
    assert first_usage == second_usage == _LIVE_USAGE
    assert memo.stats().entries == 0


def test_hit_logs_its_own_line_and_skips_the_live_call(monkeypatch, caplog):
    llm, client = _llm(monkeypatch, _response(), memo=_memo())

    with caplog.at_level("INFO", logger="search.engine.clients.llm"):
        asyncio.run(_call(llm))
        assert "LLM memo hit" not in caplog.text
        asyncio.run(_call(llm))

    assert "LLM memo hit: schema=_DummySchema" in caplog.text
    assert len(client.models.calls) == 1


def test_live_call_without_usage_metadata_reports_none_and_no_memo_hit(monkeypatch, caplog):
    """Missing `usage_metadata` does not turn a live call into a memo hit."""
    llm, client = _llm(
        monkeypatch,
        _FakeResponse(text='{"value":1}', usage_metadata=None),
        memo=_memo(),
    )

    with caplog.at_level("INFO", logger="search.engine.clients.llm"):
        first, first_usage = asyncio.run(_call(llm))
        assert "LLM memo hit" not in caplog.text
        second, second_usage = asyncio.run(_call(llm))

    assert first_usage is None
    assert first == second == _DummySchema(value=1)
    assert second_usage == _HIT_USAGE
    assert "LLM memo hit: schema=_DummySchema" in caplog.text
    assert len(client.models.calls) == 1


def test_hit_returns_a_distinct_instance_per_call(monkeypatch):
    llm, _client = _llm(monkeypatch, _response(), memo=_memo())

    first, _ = asyncio.run(_call(llm))
    second, _ = asyncio.run(_call(llm))

    assert first is not second
    first.value = 99
    third, _ = asyncio.run(_call(llm))
    assert third.value == 1


def test_parse_failure_is_not_stored(monkeypatch):
    """A response with no text raises LLMInfraError and stores
    nothing, so the next identical call reaches Gemini again."""
    llm, client = _llm(
        monkeypatch,
        _FakeResponse(text=None, usage_metadata=_FakeUsageMeta()),
        memo=_memo(),
    )

    with pytest.raises(LLMInfraError):
        asyncio.run(_call(llm))
    with pytest.raises(LLMInfraError):
        asyncio.run(_call(llm))

    assert len(client.models.calls) == 2


def test_transport_failure_is_not_stored(monkeypatch):
    llm, client = _llm(monkeypatch, RuntimeError("boom"), memo=_memo())

    with pytest.raises(LLMInfraError):
        asyncio.run(_call(llm))
    with pytest.raises(LLMInfraError):
        asyncio.run(_call(llm))

    assert len(client.models.calls) == 2


# ---------------------------------------------------------------------------
# key movement
# ---------------------------------------------------------------------------

# Every input the request resolves into, as (llm kwargs, call kwargs). A key
# that ignores one of these serves a request Gemini never saw.
_KEY_VARIANTS = {
    "prompt": ({}, {"prompt": "other prompt"}),
    "system": ({}, {"system": "other system"}),
    "temperature": ({"model": "gemini-2.5-flash"}, {}),  # pins 0.0 where 3.5+ omits it
    "thinking_level": ({"thinking_level": "high"}, {}),
    "seed": ({"seed": 99}, {}),
    "model": ({"model": "gemini-3.6-flash"}, {}),
}


@pytest.mark.parametrize("field", sorted(_KEY_VARIANTS))
def test_key_moves_on_each_request_field(monkeypatch, field):
    llm_kwargs, call_kwargs = _KEY_VARIANTS[field]
    base_call = {"prompt": "p", "system": "s", "schema": _DummySchema}
    memo = _memo()

    base_client = _CountingClient(_response())
    monkeypatch.setattr(llm_module, "_get_client", lambda: base_client)
    base_llm = LLM(memo=memo)
    asyncio.run(generate_structured_async(base_llm, **base_call))

    other_client = _CountingClient(_response())
    monkeypatch.setattr(llm_module, "_get_client", lambda: other_client)
    varied_llm = LLM(memo=memo, **llm_kwargs)
    asyncio.run(generate_structured_async(varied_llm, **{**base_call, **call_kwargs}))

    assert len(base_client.models.calls) == 1
    assert len(other_client.models.calls) == 1


def test_lone_surrogate_prompt_keys_and_behaves_as_without_a_memo(monkeypatch):
    """A prompt holding a lone surrogate, which RFC 8785 rejects, still keys: the
    ASCII-escaped memo key lets the first call reach the model and the repeat hit the memo."""
    prompt = "\ud800"
    memo = _memo()

    unmemoised, _unmemoised_client = _llm(monkeypatch, _response())
    plain, plain_usage = asyncio.run(_call(unmemoised, prompt=prompt))

    memoised, memoised_client = _llm(monkeypatch, _response(), memo=memo)
    first, first_usage = asyncio.run(_call(memoised, prompt=prompt))
    second, second_usage = asyncio.run(_call(memoised, prompt=prompt))

    assert len(memoised_client.models.calls) == 1
    assert memoised_client.models.calls[0][1] == prompt
    assert first == plain
    assert first_usage == plain_usage == _LIVE_USAGE
    assert second == first
    assert second_usage == _HIT_USAGE


def test_key_moves_on_schema(monkeypatch):
    llm, client = _llm(monkeypatch, _response(), memo=_memo())

    asyncio.run(_call(llm, schema=_DummySchema))
    asyncio.run(_call(llm, schema=_OtherSchema))

    assert len(client.models.calls) == 2


def test_key_covers_every_field_of_a_real_config():
    """The key hashes every JSON field in `GenerateContentConfig`.
    The schema's qualified name replaces `response_schema`."""
    config = GenerateContentConfig(
        system_instruction="s",
        response_mime_type="application/json",
        response_schema=_DummySchema,
        temperature=None,
        thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.LOW),
        seed=1,
    )

    dumped = config.model_dump(mode="json", exclude={"response_schema"})

    # Every field of the config is in the dump but the one deliberately excluded.
    assert set(dumped) == set(GenerateContentConfig.model_fields) - {"response_schema"}
    # The dump loses nothing: reconstructing a config from it dumps identically.
    rebuilt = GenerateContentConfig(**dumped, response_schema=_DummySchema)
    assert rebuilt.model_dump(mode="json", exclude={"response_schema"}) == dumped

    # Any one scalar field's value moving moves the key.
    for field, value in dumped.items():
        if not isinstance(value, (str, int, float, bool)):
            continue
        moved = value + 1 if isinstance(value, (int, float)) else f"{value}x"
        perturbed = GenerateContentConfig(**{**dumped, field: moved}, response_schema=_DummySchema)
        assert _memo_key(model="m", prompt="p", schema=_DummySchema, config=config) != _memo_key(
            model="m", prompt="p", schema=_DummySchema, config=perturbed
        ), field

    # The thinking config is the one nested field a raw_text resolve sets.
    other_level = GenerateContentConfig(
        **{**dumped, "thinking_config": ThinkingConfig(thinking_level=ThinkingLevel.HIGH)},
        response_schema=_DummySchema,
    )
    assert _memo_key(model="m", prompt="p", schema=_DummySchema, config=config) != _memo_key(
        model="m", prompt="p", schema=_DummySchema, config=other_level
    )


def test_a_field_added_without_a_rename_moves_the_key():
    grown = create_model("_DummySchema", __base__=_DummySchema, extra=(int, 0))
    grown.__module__ = _DummySchema.__module__
    config = GenerateContentConfig(response_mime_type="application/json")

    assert grown.__qualname__ == _DummySchema.__qualname__
    assert _memo_key(model="m", prompt="p", schema=_DummySchema, config=config) != _memo_key(
        model="m", prompt="p", schema=grown, config=config
    )


def test_the_key_is_built_from_the_config_the_call_sends(monkeypatch):
    """The memo key and `_fetch_text` use the same resolved config."""
    llm, client = _llm(monkeypatch, _response(), memo=_memo(), seed=1234)

    asyncio.run(_call(llm))

    _model, _contents, sent = client.models.calls[0]
    resolved = llm._resolve_config(system="s", schema=_DummySchema)
    assert sent.model_dump(mode="json", exclude={"response_schema"}) == resolved.model_dump(
        mode="json", exclude={"response_schema"}
    )
    assert sent.seed == 1234


# ---------------------------------------------------------------------------
# TTL and byte budget
# ---------------------------------------------------------------------------


def test_expired_entry_calls_again(monkeypatch):
    llm, client = _llm(monkeypatch, _response(), memo=_memo(ttl_s=0.005))

    asyncio.run(_call(llm))
    time.sleep(0.02)
    asyncio.run(_call(llm))

    assert len(client.models.calls) == 2


def test_entries_over_the_byte_budget_evict_least_recently_used(monkeypatch):
    """Each `_DummySchema(value=1)` answer is 11 bytes, so a 22-byte budget
    holds exactly two; the third evicts the least recently used."""
    memo = _memo(max_bytes=2 * _ANSWER_BYTES, max_entry_bytes=2 * _ANSWER_BYTES)

    # The budget in bytes, not the answer count: the memo's own store is what
    # this test sizes, so a rule that counted entries would fail here.
    assert len(_DummySchema(value=1).model_dump_json().encode()) == _ANSWER_BYTES

    llm, client = _llm(monkeypatch, _response(), memo=memo)

    for prompt in ("a", "b"):
        asyncio.run(_call(llm, prompt=prompt))
    asyncio.run(_call(llm, prompt="a"))  # 'a' now most recent
    asyncio.run(_call(llm, prompt="c"))  # evicts 'b'

    assert memo.stats().evictions == 1
    assert memo.stats().entries == 2
    assert memo.stats().total_bytes == 2 * _ANSWER_BYTES

    asyncio.run(_call(llm, prompt="a"))  # hit
    asyncio.run(_call(llm, prompt="c"))  # hit
    asyncio.run(_call(llm, prompt="b"))  # miss, refetched

    assert len(client.models.calls) == 4


def test_an_answer_over_the_per_entry_ceiling_is_served_but_not_stored(monkeypatch):
    """An answer over SEARCH_LLM_MEMO_MAX_ENTRY_BYTES still reaches the caller
    that fetched it, with its live usage, and is simply not kept: the next
    identical request is a miss rather than a serve."""
    memo = _memo(max_bytes=1024, max_entry_bytes=_ANSWER_BYTES - 1)
    llm, client = _llm(monkeypatch, _response(), memo=memo)

    served, usage = asyncio.run(_call(llm))

    assert served == _DummySchema(value=1)
    assert usage == _LIVE_USAGE
    assert memo.stats().oversized == 1
    assert memo.stats().entries == 0

    asyncio.run(_call(llm))

    assert len(client.models.calls) == 2


def test_build_memo_takes_its_bounds_from_config(search_settings):
    search_settings(llm_memo_max_bytes=4096, llm_memo_max_entry_bytes=512, llm_memo_ttl_s=12.0)

    memo = build_memo()

    assert memo is not None
    assert (memo._max_bytes, memo._max_entry_bytes, memo._ttl_s) == (4096, 512, 12.0)


# ---------------------------------------------------------------------------
# single-flight on the event loop
# ---------------------------------------------------------------------------


class _BlockingModels(_CountingModels):
    """Holds the live call until `release`; both events run on the leader's event loop."""

    def __init__(
        self,
        answer,
        *,
        started: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        super().__init__(answer)
        self._started = started
        self._release = release

    async def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        self._started.set()
        try:
            await asyncio.wait_for(self._release.wait(), timeout=5)
        except TimeoutError as exc:
            raise AssertionError("test deadlocked waiting for release") from exc
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _blocking(monkeypatch, *, answer=None):
    """An LLM whose one live call blocks, plus its (started, release) pair."""
    started = asyncio.Event()
    release = asyncio.Event()
    client = _CountingClient(answer if answer is not None else _response())
    client.models = _BlockingModels(client.models.answer, started=started, release=release)
    client.aio = SimpleNamespace(models=client.models)
    monkeypatch.setattr(llm_module, "_get_client", lambda: client)
    return LLM(memo=_memo()), client, started, release


async def _parked(memo: LLMMemo, *, misses: int) -> None:
    """Wait until `misses` callers claim the key.
    Later misses are followers parked on the leader."""
    for _ in range(500):
        if memo.stats().misses >= misses:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"only {memo.stats().misses} of {misses} callers claimed the key")


def test_concurrent_identical_calls_make_one_api_call(monkeypatch):
    """Concurrent identical calls share the leader's live fetch."""

    async def scenario():
        llm, client, started, release = _blocking(monkeypatch)
        memo = llm._memo
        assert memo is not None

        leader = asyncio.create_task(_call(llm))
        await started.wait()
        follower = asyncio.create_task(_call(llm))
        await _parked(memo, misses=2)

        release.set()
        results = await asyncio.gather(leader, follower)
        return client, results

    client, results = asyncio.run(scenario())

    assert len(client.models.calls) == 1
    assert [r[0] for r in results] == [_DummySchema(value=1)] * 2
    # The leader reports the real spend; the follower, served from the memo,
    # reports zeros.
    assert sorted(r[1]["input_tokens"] for r in results) == [0, 11]


def test_leader_failure_reaches_the_follower_and_stores_nothing(monkeypatch):
    async def scenario():
        llm, client, started, release = _blocking(monkeypatch, answer=RuntimeError("boom"))
        memo = llm._memo
        assert memo is not None

        leader = asyncio.create_task(_call(llm))
        await started.wait()
        follower = asyncio.create_task(_call(llm))
        await _parked(memo, misses=2)

        release.set()
        errors = []
        for task in (leader, follower):
            try:
                await task
            except BaseException as exc:
                errors.append(exc)
        return client, errors

    client, errors = asyncio.run(scenario())

    assert len(errors) == 2
    assert all(isinstance(exc, LLMInfraError) for exc in errors)
    assert "boom" in str(errors[0])
    assert len(client.models.calls) == 1  # the follower did not retry


def test_a_follower_whose_wait_outlives_the_timeout_raises_llminfraerror(
    monkeypatch, search_settings
):
    """A follower timeout raises `LLMInfraError`; the leader continues and caches its result."""
    search_settings(llm_timeout_s=0.05)

    async def scenario():
        llm, client, started, release = _blocking(monkeypatch)
        memo = llm._memo
        assert memo is not None

        leader = asyncio.create_task(_call(llm))
        await started.wait()
        follower = asyncio.create_task(_call(llm))
        await _parked(memo, misses=2)

        with pytest.raises(LLMInfraError) as caught:
            await follower
        assert "SEARCH_LLM_TIMEOUT_S" in str(caught.value)

        release.set()
        leader_result = await leader
        return client, memo, leader_result

    client, memo, leader_result = asyncio.run(scenario())

    assert leader_result == (_DummySchema(value=1), _LIVE_USAGE)
    assert len(client.models.calls) == 1  # the follower did not retry
    # The leader's result landed anyway, so the next identical call is a hit.
    assert memo.stats().entries == 1


def test_cancelling_the_leaders_request_still_publishes_for_followers(monkeypatch):
    """SSE cancellation detaches its caller while the separate fetch serves parked followers."""

    async def scenario():
        llm, client, started, release = _blocking(monkeypatch)
        memo = llm._memo
        assert memo is not None

        leader = asyncio.create_task(_call(llm))
        await started.wait()
        follower = asyncio.create_task(_call(llm))
        await _parked(memo, misses=2)

        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader

        release.set()
        follower_result = await follower
        return llm, client, memo, follower_result

    llm, client, memo, follower_result = asyncio.run(scenario())

    assert follower_result == (_DummySchema(value=1), _HIT_USAGE)
    assert len(client.models.calls) == 1
    # Published, not abandoned: a later request is a stored hit.
    assert memo.stats().entries == 1
    assert asyncio.run(_call(llm))[0] == _DummySchema(value=1)
    assert len(client.models.calls) == 1
    assert set() == llm_module._INFLIGHT_LEADERS


def test_cancelling_a_follower_leaves_the_leader_unaffected(monkeypatch):
    async def scenario():
        llm, client, started, release = _blocking(monkeypatch)
        memo = llm._memo
        assert memo is not None

        leader = asyncio.create_task(_call(llm))
        await started.wait()
        follower = asyncio.create_task(_call(llm))
        await _parked(memo, misses=2)

        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower

        release.set()
        leader_result = await leader
        return llm, client, leader_result

    llm, client, leader_result = asyncio.run(scenario())

    assert leader_result == (_DummySchema(value=1), _LIVE_USAGE)
    assert len(client.models.calls) == 1
    # The leader's own answer is stored and served to the next caller.
    assert asyncio.run(_call(llm))[1] == _HIT_USAGE
    assert len(client.models.calls) == 1


def test_leader_publish_failure_still_returns_the_answer_and_frees_the_key(
    monkeypatch, search_settings
):
    """A publish failure preserves the leader's answer and promptly fails its follower.
    The key is free for a fresh fetch."""
    search_settings(llm_timeout_s=0.2)

    async def scenario():
        llm, client, started, release = _blocking(monkeypatch)
        memo = llm._memo
        assert memo is not None
        real_publish = memo.publish

        async def _boom_publish(*args, **kwargs):
            raise RuntimeError("publish boom")

        monkeypatch.setattr(memo, "publish", _boom_publish)

        leader = asyncio.create_task(_call(llm))
        await started.wait()
        follower = asyncio.create_task(_call(llm))
        await _parked(memo, misses=2)

        release.set()
        leader_result = await leader
        follower_start = time.monotonic()
        with pytest.raises(LLMInfraError) as caught:
            await follower
        follower_elapsed = time.monotonic() - follower_start
        # A prompt outcome from abandon(), not a stall past the wait_for
        # deadline: the failure detail names the leader failing, and the wait
        # finishes well inside SEARCH_LLM_TIMEOUT_S rather than against it.
        assert "shared in-flight Gemini" in str(caught.value)
        assert follower_elapsed < 0.1

        assert memo.stats().entries == 0  # publish raised, so nothing stored

        monkeypatch.setattr(memo, "publish", real_publish)
        later_result = await _call(llm)
        return client, leader_result, later_result

    client, leader_result, later_result = asyncio.run(scenario())

    assert leader_result == (_DummySchema(value=1), _LIVE_USAGE)
    assert later_result[0] == _DummySchema(value=1)
    # The fresh leader made its own Gemini call rather than timing out as a
    # follower on a wedged key.
    assert len(client.models.calls) == 2


class _HeldModels(_CountingModels):
    """Holds the first live call open until `release` is set, and records
    whether that call was cancelled. Every later call answers at once."""

    def __init__(self, answer) -> None:
        super().__init__(answer)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        if len(self.calls) > 1:
            return self.answer
        self.started.set()
        try:
            await asyncio.wait_for(self.release.wait(), timeout=5)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        except TimeoutError as exc:
            raise AssertionError("test deadlocked waiting for release") from exc
        return self.answer


def _held(monkeypatch) -> tuple[LLM, LLMMemo, _HeldModels]:
    """An LLM with a memo whose first live call is held open."""
    models = _HeldModels(_response())
    client = SimpleNamespace(models=models, aio=SimpleNamespace(models=models))
    monkeypatch.setattr(llm_module, "_get_client", lambda: client)
    memo = _memo()
    return LLM(memo=memo), memo, models


async def _settled(models: _HeldModels) -> None:
    """Wait until the held call has seen its cancellation."""
    await asyncio.wait_for(models.cancelled.wait(), timeout=2)


def test_cancelling_an_orphaned_leader_cancels_its_call_and_frees_the_key(monkeypatch, caplog):
    """The leader's caller leaves with nobody parked on the key, so the live
    call is cancelled, logged once by schema name, nothing is stored, and the
    next caller makes its own call."""
    caplog.set_level("INFO", logger=llm_module.__name__)

    async def scenario():
        llm, memo, models = _held(monkeypatch)
        leader = asyncio.create_task(_call(llm))
        await models.started.wait()

        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        await _settled(models)

        assert memo.stats().entries == 0
        return await _call(llm)

    later = asyncio.run(scenario())

    assert later == (_DummySchema(value=1), _LIVE_USAGE)
    assert set() == llm_module._INFLIGHT_LEADERS
    cancelled = [r.getMessage() for r in caplog.records if "orphaned" in r.getMessage()]
    assert cancelled == ["LLM memo cancelled an orphaned Gemini call: schema=_DummySchema"]


def test_cancelling_the_leader_with_a_follower_parked_keeps_the_call(monkeypatch):
    async def scenario():
        llm, memo, models = _held(monkeypatch)
        leader = asyncio.create_task(_call(llm))
        await models.started.wait()
        follower = asyncio.create_task(_call(llm))
        await _parked(memo, misses=2)

        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        await asyncio.sleep(0.05)
        assert not models.cancelled.is_set()

        models.release.set()
        return memo, models, await follower

    memo, models, follower_result = asyncio.run(scenario())

    assert follower_result == (_DummySchema(value=1), _HIT_USAGE)
    assert not models.cancelled.is_set()
    assert len(models.calls) == 1
    assert memo.stats().entries == 1


@pytest.mark.parametrize("departure", ["cancelled", "timed_out"])
def test_the_last_follower_leaving_after_the_leader_cancels_the_call(
    monkeypatch, search_settings, departure: str
):
    """The leader's caller has gone, and the one follower then leaves too,
    by cancellation or by its own SEARCH_LLM_TIMEOUT_S deadline: nobody is
    waiting, so the call is cancelled and the key is free."""
    search_settings(llm_timeout_s=0.3)

    async def scenario():
        llm, memo, models = _held(monkeypatch)
        leader = asyncio.create_task(_call(llm))
        await models.started.wait()
        follower = asyncio.create_task(_call(llm))
        await _parked(memo, misses=2)

        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        assert not models.cancelled.is_set()

        if departure == "cancelled":
            follower.cancel()
            with pytest.raises(asyncio.CancelledError):
                await follower
        else:
            with pytest.raises(LLMInfraError, match="SEARCH_LLM_TIMEOUT_S"):
                await follower
        await _settled(models)

        assert memo.stats().entries == 0
        return models, await _call(llm)

    models, later = asyncio.run(scenario())

    assert later == (_DummySchema(value=1), _LIVE_USAGE)
    assert len(models.calls) == 2
    assert set() == llm_module._INFLIGHT_LEADERS


def test_a_caller_arriving_during_the_orphans_cancellation_leads_its_own_call(
    monkeypatch, search_settings
):
    """A newcomer can lead after orphan detachment but before the old fetch handles cancellation.
    The old fetch's late abandon must leave the new flight intact."""
    search_settings(llm_timeout_s=1.0)

    async def scenario():
        llm, memo, models = _held(monkeypatch)
        real_acquire = memo.acquire
        # (is_leader, held call already cancelled) for every miss.
        claims: list[tuple[bool, bool]] = []

        async def _recording_acquire(key):
            value, lease = await real_acquire(key)
            if lease is not None:
                claims.append((lease.is_leader, models.cancelled.is_set()))
            return value, lease

        monkeypatch.setattr(memo, "acquire", _recording_acquire)
        leader = asyncio.create_task(_call(llm))
        await models.started.wait()

        leader.cancel()
        newcomer = asyncio.create_task(_call(llm))
        with pytest.raises(asyncio.CancelledError):
            await leader
        result = await newcomer
        await _settled(models)
        return memo, models, claims, result

    memo, models, claims, result = asyncio.run(scenario())

    assert result == (_DummySchema(value=1), _LIVE_USAGE)
    # The newcomer claimed the key as a leader while the held call was still
    # running, so the two flights overlapped.
    assert claims == [(True, False), (True, False)]
    assert len(models.calls) == 2
    assert memo.stats().entries == 1
    assert set() == llm_module._INFLIGHT_LEADERS


# ---------------------------------------------------------------------------
# an answer the schema refuses
# ---------------------------------------------------------------------------

# A binding that names no value: `_UniversalBindOutput` refuses it with `SCHEMA_REFUSAL`.
_REFUSED_BIND = json.dumps(
    {
        "chosen_shape_index": 0,
        "bindings": [
            {"axis": "how", "property_dcid": "gender", "kind": "value", "value_dcids": []}
        ],
    }
)


def test_a_refused_answer_reaches_leader_follower_and_hit_as_a_refusal(monkeypatch):
    """The memo stores the model's text and every holder parses it alike: the leader's
    refusal carries the live usage, the follower's and a later hit's carry zero usage,
    and none of them calls the model again."""
    refused = _FakeResponse(text=_REFUSED_BIND, usage_metadata=_FakeUsageMeta())

    async def scenario():
        llm, client, started, release = _blocking(monkeypatch, answer=refused)
        memo = llm._memo
        assert memo is not None

        leader = asyncio.create_task(_call(llm, schema=_UniversalBindOutput))
        await started.wait()
        follower = asyncio.create_task(_call(llm, schema=_UniversalBindOutput))
        await _parked(memo, misses=2)

        release.set()
        outcomes = await asyncio.gather(leader, follower, return_exceptions=True)
        hit = await asyncio.gather(_call(llm, schema=_UniversalBindOutput), return_exceptions=True)
        return client, [*outcomes, *hit]

    client, outcomes = asyncio.run(scenario())

    assert [type(exc) for exc in outcomes] == [LLMOutputRefused] * 3
    assert [exc.usage for exc in outcomes] == [_LIVE_USAGE, _HIT_USAGE, _HIT_USAGE]
    assert len(client.models.calls) == 1


def test_round_trip_holds_through_a_memo_hit(monkeypatch):
    """A realistic extraction fetched once and then served from the memo
    compares equal to the first."""
    instance = Extraction.model_validate(_fixture("Extraction"))
    llm, client = _llm(
        monkeypatch,
        _FakeResponse(text=instance.model_dump_json(), usage_metadata=_FakeUsageMeta()),
        memo=_memo(),
    )

    first, _ = asyncio.run(_call(llm, schema=Extraction))
    second, usage = asyncio.run(_call(llm, schema=Extraction))

    assert len(client.models.calls) == 1
    assert second == instance
    assert second.model_dump_json() == first.model_dump_json()
    assert usage == _HIT_USAGE


# ---------------------------------------------------------------------------
# the stage clone
# ---------------------------------------------------------------------------


def test_for_stage_clone_carries_the_memo(monkeypatch):
    """Stage clones share the memo, while the pinned thinking level gives clone and parent separate keys."""
    memo = _memo()
    llm, client = _llm(monkeypatch, _response(), memo=memo)

    first_clone = for_stage(llm, "bind")
    second_clone = for_stage(llm, "bind")
    assert first_clone is not llm and second_clone is not llm
    assert first_clone._memo is memo  # ty: ignore[unresolved-attribute]  # for_stage clones the LLM

    asyncio.run(_call(first_clone))
    asyncio.run(_call(second_clone))

    assert len(client.models.calls) == 1


def test_for_stage_clone_keys_apart_from_its_parent(monkeypatch):
    memo = _memo()
    llm, client = _llm(monkeypatch, _response(), memo=memo)

    asyncio.run(_call(for_stage(llm, "bind")))
    asyncio.run(_call(llm))

    assert len(client.models.calls) == 2
    # Parent and clone are separate memo entries, so neither evicted the other.
    assert memo.stats().entries == 2


class _AnsweringThenRefusing:
    """An LLM double answering its first call and refusing every later one, each with the
    usage `_LIVE_USAGE`."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(self, *, prompt, system, schema):
        self.calls += 1
        if self.calls > 1:
            raise LLMOutputRefused("refused", usage=dict(_LIVE_USAGE))
        return schema(value=1), dict(_LIVE_USAGE)


def test_the_usage_record_holds_every_call_inside_it_a_refused_one_included():
    """`recording_llm_usage` collects the usage of each call made inside it, an answer the
    schema refuses included, and its enclosing record gains them when it closes."""
    llm = _AnsweringThenRefusing()

    async def calls() -> None:
        await generate_structured_async(llm, prompt="p", system="s", schema=_DummySchema)
        with pytest.raises(LLMOutputRefused):
            await generate_structured_async(llm, prompt="p", system="s", schema=_DummySchema)

    with recording_llm_usage() as outer, recording_llm_usage() as inner:
        asyncio.run(calls())

    assert inner.calls == [_LIVE_USAGE, _LIVE_USAGE]
    assert outer.calls == inner.calls
