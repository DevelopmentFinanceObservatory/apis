"""`jev_client.ask` over an injected `httpx2.MockTransport`: the retry rule per status,
`Retry-After` against the remaining budget, the ceiling, the circuit breaker, split
requests under one deadline and the concurrency semaphore, and the recorded wire contract."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import httpx2
import pytest
from backports import zstd
from pydantic import SecretStr

from search.engine import config
from search.engine.binders import jev_client
from search.engine.binders.jev_client import (
    BREAKER_FAILURES,
    BREAKER_OPEN_S,
    BREAKER_WINDOW_S,
    CircuitBreaker,
    JevError,
    JevResponse,
    ask,
    build_client,
)
from search.engine.binders.jev_split import split_questions
from search.engine.clients.loop_local import LoopLocal

type Handler = Callable[[httpx2.Request], Awaitable[httpx2.Response]]

_FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "jev" / "system_one_response.json").read_text()
)
_STATE = {"query": "life expectancy at birth in Kenya"}
_QUESTIONS = {"control": {"type": "noul", "instructions": "Does the query name a country?"}}


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _FrozenLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock moves only through `advance`, so a timer fires when a test
    says the loop stalled past it and never because the machine was slow. `advance` called
    from a handler is a loop that stalled for that long: nothing runs until it returns, and
    the next step finds every timer due by then."""

    def __init__(self) -> None:
        super().__init__()
        self._frozen_now = 0.0

    def time(self) -> float:
        return self._frozen_now

    def advance(self, seconds: float) -> None:
        self._frozen_now += seconds


class _Jev:
    """The module's client, breaker and semaphores, rebuilt over a scripted handler."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self.clock = _Clock()
        self.requests: list[dict] = []
        self.settings = config.settings().model_copy(
            update={"jev_api_key": SecretStr("test-key"), "jev_ceiling_s": 2.5}
        )
        monkeypatch.setattr(jev_client, "_BREAKER", CircuitBreaker(clock=self.clock))
        monkeypatch.setattr(jev_client, "_SEMAPHORES", LoopLocal())

    def serve(self, handler: Handler, **settings: object) -> None:
        self.settings = self.settings.model_copy(update=settings)

        async def recording(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(_body(request))
            return await handler(request)

        transport = httpx2.MockTransport(recording)
        self._monkeypatch.setattr(
            jev_client, "_CLIENT", build_client(self.settings, transport=transport)
        )

    def ask(
        self,
        questions: dict = _QUESTIONS,
        state: object = _STATE,
        *,
        loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None,
    ) -> JevResponse:
        return asyncio.run(ask(state, questions, settings=self.settings), loop_factory=loop_factory)


@pytest.fixture
def jev(monkeypatch: pytest.MonkeyPatch) -> _Jev:
    return _Jev(monkeypatch)


def _body(request: httpx2.Request) -> dict:
    content = request.content
    if request.headers.get("content-encoding") == "zstd":
        content = zstd.decompress(content)
    return json.loads(content)


def _answers(request: httpx2.Request) -> httpx2.Response:
    """A 200 answering every noul in the request with 0.9."""
    questions = _body(request)["questions"]
    return httpx2.Response(
        200,
        json={
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "answers": {qid: {"type": "noul", "noul": 0.9} for qid in questions},
        },
    )


def _script(*statuses: int, headers: dict[str, str] | None = None) -> Handler:
    """Answer the n-th request with the n-th status, then 200s."""
    remaining = list(statuses)

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if remaining:
            return httpx2.Response(
                remaining.pop(0), json={"error": "scripted"}, headers=headers or {}
            )
        return _answers(request)

    return handler


# Retry-After-Ms 0 skips the backoff sleep so a retried status costs no wall time.
_NOW = {"retry-after-ms": "0"}


@pytest.mark.parametrize("status", [429, 500, 503, 529])
def test_rate_limit_and_server_errors_retry_once(jev: _Jev, status: int) -> None:
    jev.serve(_script(status, headers=_NOW))
    assert jev.ask().nouls == {"control": 0.9}
    assert len(jev.requests) == 2


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "rejected"),
        (401, "auth"),
        (402, "payment_required"),
        (403, "auth"),
        (408, "request_timeout"),
        (422, "rejected"),
    ],
)
def test_other_statuses_fail_without_a_retry(jev: _Jev, status: int, kind: str) -> None:
    jev.serve(_script(status, headers=_NOW))
    with pytest.raises(JevError) as raised:
        jev.ask()
    assert (raised.value.status, raised.value.kind) == (status, kind)
    assert len(jev.requests) == 1


def test_a_second_server_error_fails_the_call(jev: _Jev) -> None:
    jev.serve(_script(503, 503, headers=_NOW))
    with pytest.raises(JevError) as raised:
        jev.ask()
    assert (raised.value.status, raised.value.kind) == (503, "server")
    assert len(jev.requests) == 2


@pytest.mark.parametrize(
    "error",
    [httpx2.ReadError("connection reset by peer"), httpx2.ConnectError("connection refused")],
)
def test_a_failed_connection_retries_once(jev: _Jev, error: Exception) -> None:
    # The retry waits the SDK's first backoff, about 0.4 s, since no Retry-After came back.
    failures = [error]

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if failures:
            raise failures.pop()
        return _answers(request)

    jev.serve(handler)
    assert jev.ask().nouls == {"control": 0.9}
    assert len(jev.requests) == 2


def test_a_second_failed_connection_fails_the_call(jev: _Jev) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadError("connection reset by peer")

    jev.serve(handler)
    with pytest.raises(JevError) as raised:
        jev.ask()
    assert (raised.value.status, raised.value.kind) == (None, "connection")
    assert len(jev.requests) == 2


def test_a_timed_out_attempt_fails_without_a_retry(jev: _Jev) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow")

    jev.serve(handler)
    with pytest.raises(JevError) as raised:
        jev.ask()
    assert (raised.value.status, raised.value.kind) == (None, "timeout")
    assert len(jev.requests) == 1


def test_retry_after_inside_the_budget_is_honoured(jev: _Jev) -> None:
    jev.serve(_script(429, headers={"retry-after-ms": "300"}))
    started = time.perf_counter()
    jev.ask()
    assert time.perf_counter() - started >= 0.3
    assert len(jev.requests) == 2


def test_retry_after_past_the_budget_fails_at_once(jev: _Jev) -> None:
    jev.serve(_script(429, headers={"retry-after": "5"}))
    started = time.perf_counter()
    with pytest.raises(JevError) as raised:
        jev.ask()
    assert time.perf_counter() - started < 1.0
    assert raised.value.kind == "rate_limited"
    assert len(jev.requests) == 1


def test_the_ceiling_cancels_an_attempt_in_flight(jev: _Jev) -> None:
    cancelled = []

    async def stall(request: httpx2.Request) -> httpx2.Response:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return _answers(request)

    jev.serve(stall, jev_ceiling_s=0.2)
    started = time.perf_counter()
    with pytest.raises(JevError) as raised:
        jev.ask()
    assert time.perf_counter() - started < 1.0
    assert (raised.value.status, raised.value.kind) == (None, "timeout")
    assert cancelled == [True]


def test_an_unanswered_question_fails_the_call(jev: _Jev) -> None:
    async def drop_last(request: httpx2.Request) -> httpx2.Response:
        body = _answers(request).json()
        body["answers"].pop("second")
        return httpx2.Response(200, json=body)

    jev.serve(drop_last)
    with pytest.raises(JevError) as raised:
        jev.ask(_QUESTIONS | {"second": {"type": "noul", "instructions": "Is it a rate?"}})
    assert raised.value.kind == "unanswered"


def _unhealthy(calls: int) -> Handler:
    """Fail `calls` calls with a 503 and its retry, then answer."""
    return _script(*[503] * (2 * calls), headers=_NOW)


def _out_of_credit(calls: int) -> Handler:
    """Fail `calls` calls with a 402, which is not retried, then answer."""
    return _script(*[402] * calls)


def _timing_out(calls: int) -> Handler:
    """Fail `calls` calls on the request's own read timeout, which is not retried, then
    answer."""
    remaining = [calls]

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if remaining[0]:
            remaining[0] -= 1
            raise httpx2.ReadTimeout("slow")
        return _answers(request)

    return handler


def _fail(jev: _Jev, times: int) -> None:
    for _ in range(times):
        with pytest.raises(JevError):
            jev.ask()


@pytest.mark.parametrize("failing", [_unhealthy, _out_of_credit, _timing_out])
def test_breaker_opens_after_consecutive_failures_and_a_probe_closes_it(
    jev: _Jev, failing: Callable[[int], Handler]
) -> None:
    jev.serve(failing(BREAKER_FAILURES))
    _fail(jev, BREAKER_FAILURES)
    sent = len(jev.requests)

    with pytest.raises(JevError) as raised:
        jev.ask()
    assert raised.value.kind == "breaker_open"
    assert len(jev.requests) == sent

    jev.clock.now += BREAKER_OPEN_S
    assert jev.ask().nouls == {"control": 0.9}
    assert jev.ask().nouls == {"control": 0.9}
    assert len(jev.requests) == sent + 2


def test_a_failed_probe_opens_the_breaker_again(jev: _Jev) -> None:
    jev.serve(_unhealthy(BREAKER_FAILURES + 1))
    _fail(jev, BREAKER_FAILURES)
    jev.clock.now += BREAKER_OPEN_S
    _fail(jev, 1)
    sent = len(jev.requests)

    jev.clock.now += BREAKER_OPEN_S - 1
    with pytest.raises(JevError) as raised:
        jev.ask()
    assert raised.value.kind == "breaker_open"
    assert len(jev.requests) == sent


def test_failures_spread_past_the_window_leave_the_breaker_closed(jev: _Jev) -> None:
    jev.serve(_unhealthy(BREAKER_FAILURES))
    _fail(jev, BREAKER_FAILURES - 1)
    jev.clock.now += BREAKER_WINDOW_S + 1
    _fail(jev, 1)
    assert jev.ask().nouls == {"control": 0.9}


def test_a_success_resets_the_consecutive_count(jev: _Jev) -> None:
    jev.serve(_unhealthy(BREAKER_FAILURES - 1))
    _fail(jev, BREAKER_FAILURES - 1)
    jev.ask()
    jev.serve(_unhealthy(BREAKER_FAILURES - 1))
    _fail(jev, BREAKER_FAILURES - 1)
    assert jev.ask().nouls == {"control": 0.9}


async def _unanswered(request: httpx2.Request) -> httpx2.Response:
    body = _answers(request).json()
    body["answers"].clear()
    return httpx2.Response(200, json=body)


async def _rejected(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(400, json={"error": "scripted"})


@pytest.mark.parametrize(
    ("handler", "kind"), [(_rejected, "rejected"), (_unanswered, "unanswered")]
)
def test_a_failure_particular_to_the_request_leaves_the_breaker_closed(
    jev: _Jev, handler: Handler, kind: str
) -> None:
    jev.serve(handler)
    for _ in range(BREAKER_FAILURES + 1):
        with pytest.raises(JevError) as raised:
            jev.ask()
        assert raised.value.kind == kind

    jev.serve(_script())
    assert jev.ask().nouls == {"control": 0.9}


def test_a_missing_key_leaves_the_breaker_closed(
    jev: _Jev, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jev_client, "_CLIENT", None)
    keyless = jev.settings.model_copy(update={"jev_api_key": SecretStr("")})
    for _ in range(BREAKER_FAILURES + 1):
        with pytest.raises(JevError, match="no_key"):
            asyncio.run(ask(_STATE, _QUESTIONS, settings=keyless))

    jev.serve(_script())
    assert jev.ask().nouls == {"control": 0.9}


def test_a_call_that_never_took_a_slot_fails_as_slot_wait_and_leaves_the_breaker_closed(
    jev: _Jev,
) -> None:
    """Every slot is held past the ceiling, so each call times out before it sends: it fails
    under its own kind and the breaker, which counts only Jev's own failures, stays shut."""
    jev.serve(_script(), jev_max_concurrency=1, jev_ceiling_s=0.05)

    async def run() -> list[BaseException | JevResponse]:
        semaphore = jev_client._SEMAPHORES.get(lambda: asyncio.Semaphore(1))
        async with semaphore:
            waits = await asyncio.gather(
                *(ask(_STATE, _QUESTIONS, settings=jev.settings) for _ in range(BREAKER_FAILURES)),
                return_exceptions=True,
            )
        return [*waits, await ask(_STATE, _QUESTIONS, settings=jev.settings)]

    *waits, answered = asyncio.run(run())
    assert [getattr(w, "kind", w) for w in waits] == ["slot_wait"] * BREAKER_FAILURES
    assert isinstance(answered, JevResponse)
    assert len(jev.requests) == 1


def test_a_call_admitted_late_and_cut_by_the_ceiling_leaves_the_breaker_closed(
    jev: _Jev,
) -> None:
    """Each call queues for most of its ceiling, then sends a request that stalls: the
    ceiling cuts a request sent after a wait for its slot, so the call fails as
    ``queued_timeout``, the breaker stays shut and the next call answers."""
    stalls = [BREAKER_FAILURES]

    async def stall_then_answer(request: httpx2.Request) -> httpx2.Response:
        if stalls[0]:
            stalls[0] -= 1
            await asyncio.sleep(30)
        return _answers(request)

    jev.serve(stall_then_answer, jev_max_concurrency=1, jev_ceiling_s=0.2)

    async def late(semaphore: asyncio.Semaphore) -> BaseException | JevResponse:
        await semaphore.acquire()
        call = asyncio.create_task(ask(_STATE, _QUESTIONS, settings=jev.settings))
        await asyncio.sleep(0.15)
        semaphore.release()
        return (await asyncio.gather(call, return_exceptions=True))[0]

    async def run() -> list[BaseException | JevResponse]:
        semaphore = jev_client._SEMAPHORES.get(lambda: asyncio.Semaphore(1))
        waits = [await late(semaphore) for _ in range(BREAKER_FAILURES)]
        return [*waits, await ask(_STATE, _QUESTIONS, settings=jev.settings)]

    *waits, answered = asyncio.run(run())
    assert [getattr(w, "kind", w) for w in waits] == ["queued_timeout"] * BREAKER_FAILURES
    assert isinstance(answered, JevResponse)
    assert len(jev.requests) == BREAKER_FAILURES + 1


def test_an_answer_in_hand_when_the_ceiling_fires_is_returned(jev: _Jev) -> None:
    """A loop stalled past the ceiling runs the response in the step before the ceiling's
    timer, so the request has answered when the ceiling fires and nothing is left to cut."""

    async def stalled(request: httpx2.Request) -> httpx2.Response:
        time.sleep(0.3)
        return _answers(request)

    jev.serve(stalled, jev_ceiling_s=0.2)
    assert jev.ask().nouls == {"control": 0.9}
    assert len(jev.requests) == 1


def test_only_one_probe_runs_while_the_breaker_is_half_open() -> None:
    clock = _Clock()
    breaker = CircuitBreaker(clock=clock)
    for _ in range(BREAKER_FAILURES):
        breaker.leave(False, probe=breaker.enter())
    clock.now += BREAKER_OPEN_S

    assert breaker.enter() is True
    with pytest.raises(JevError, match="breaker_open"):
        breaker.enter()
    breaker.leave(None, probe=True)
    assert breaker.enter() is True


def _large_questions() -> dict[str, dict[str, str]]:
    return {f"m|{i}": {"type": "noul", "instructions": f"{i} " + "x" * 9_000} for i in range(24)}


def test_a_split_request_merges_every_chunk(jev: _Jev) -> None:
    async def answer(request: httpx2.Request) -> httpx2.Response:
        return _answers(request)

    jev.serve(answer)
    questions = _large_questions()
    response = jev.ask(questions)

    assert len(jev.requests) > 1
    assert all(r["state"] == _STATE for r in jev.requests)
    assert sorted(q for r in jev.requests for q in r["questions"]) == sorted(questions)
    assert response.nouls == dict.fromkeys(questions, 0.9)
    assert response.usage.input_tokens == 10 * len(jev.requests)


def test_one_failed_chunk_fails_the_call(jev: _Jev) -> None:
    async def fail_one(request: httpx2.Request) -> httpx2.Response:
        if "m|0" in _body(request)["questions"]:
            return httpx2.Response(400, json={"error": "scripted"})
        return _answers(request)

    jev.serve(fail_one)
    with pytest.raises(JevError) as raised:
        jev.ask(_large_questions())
    assert (raised.value.status, raised.value.kind) == (400, "rejected")


def test_chunks_share_one_deadline_and_count_against_the_semaphore(jev: _Jev) -> None:
    in_flight = 0
    peak = 0

    async def slow(request: httpx2.Request) -> httpx2.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.15)
        finally:
            in_flight -= 1
        return _answers(request)

    # Four or more chunks two at a time take at least 0.3 s, past the 0.25 s ceiling. The
    # ceiling cuts the chunks that queued behind the first two, which sent after waiting
    # for their slots, so the call fails as a queued timeout.
    jev.serve(slow, jev_max_concurrency=2, jev_ceiling_s=0.25)
    with pytest.raises(JevError) as raised:
        jev.ask(_large_questions())
    assert raised.value.kind == "queued_timeout"
    assert peak == 2


def test_split_answers_in_hand_at_the_ceiling_are_returned_only_when_every_chunk_answered(
    jev: _Jev,
) -> None:
    """After a stall every chunk has answered, so the call returns them all. With one chunk
    still outstanding, the answered chunks complete nothing and the call fails. The loop's
    clock is frozen, and the handler that sees every chunk arrive moves it past the ceiling
    while the others are runnable, which is a loop stalled past the ceiling."""
    questions = _large_questions()
    chunks = len(split_questions(_STATE, questions))
    assert chunks > 2

    def serve(*, outstanding: bool) -> None:
        arrived = 0
        all_in = asyncio.Event()

        async def handler(request: httpx2.Request) -> httpx2.Response:
            nonlocal arrived
            arrived += 1
            if arrived == chunks:
                all_in.set()
                loop = cast(_FrozenLoop, asyncio.get_running_loop())
                loop.advance(jev.settings.jev_ceiling_s)
            await all_in.wait()
            if outstanding and "m|0" in _body(request)["questions"]:
                await asyncio.Event().wait()
            return _answers(request)

        jev.serve(handler, jev_ceiling_s=0.2)

    serve(outstanding=False)
    assert jev.ask(questions, loop_factory=_FrozenLoop).nouls == dict.fromkeys(questions, 0.9)
    assert len(jev.requests) == chunks

    serve(outstanding=True)
    with pytest.raises(JevError) as raised:
        jev.ask(questions, loop_factory=_FrozenLoop)
    assert raised.value.kind == "timeout"


def test_the_zstd_backport_compresses_large_bodies(jev: _Jev) -> None:
    seen = []

    async def answer(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers.get("content-encoding"))
        return _answers(request)

    jev.serve(answer)
    jev.ask({"big": {"type": "noul", "instructions": "x" * 4_000}})
    assert seen == ["zstd"]


def test_an_empty_key_fails_before_any_request(jev: _Jev) -> None:
    settings = jev.settings.model_copy(update={"jev_api_key": SecretStr("")})
    with pytest.raises(JevError) as raised:
        build_client(settings)
    assert raised.value.kind == "no_key"


def test_recorded_response_parses(jev: _Jev) -> None:
    async def recorded(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_FIXTURE["response"])

    jev.serve(recorded, jev_model=_FIXTURE["model"])
    response = jev.ask(_FIXTURE["questions"], state=_FIXTURE["state"])

    # The control asks whether "life expectancy at birth in Kenya" names a country.
    assert response.nouls["control"] > 0.9
    assert response.nouls["m|3"] < 0.1
    form = response.choices["form"]
    assert form.keys() == _FIXTURE["questions"]["form"]["criteria"].keys()
    assert max(form, key=form.__getitem__) == "level"
    assert sum(form.values()) == pytest.approx(1.0, abs=0.02)
    assert response.model == "jev-1.13.0"
    assert response.usage.input_tokens > 0
    assert jev.requests == [
        {
            "model": _FIXTURE["model"],
            "state": _FIXTURE["state"],
            "questions": _FIXTURE["questions"],
        }
    ]
