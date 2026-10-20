"""The Jev binder: the quantity read and the bind call as two Jev requests.

The quantity read sends `jev_request.quantity_request` and reads it with
`jev_assemble.assemble_quantity`, in place where the Gemini read runs, over the admissible
shapes' denominators. The bind call sends `jev_request.bind_request` over the menu and
assembles the answer with `jev_assemble.bind_result`, which runs the request's house-policy
step once on an answer that elects a shape. Every failure of a Jev request leaves as one
`JevFailure`, raised before any answer is assembled, so a binder that falls back on it runs
the house policies at most once per call.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from typesafe_sdk import JSONContent

from search.engine.binders.base import BindRequest, BindResult, QuantityRequest
from search.engine.binders.jev_assemble import (
    JevAnswers,
    assemble_quantity,
    bind_result,
    frame_stated_form,
)
from search.engine.binders.jev_client import JevError, JevQuestion, JevResponse, ask
from search.engine.binders.jev_menu import jev_menu
from search.engine.binders.jev_request import bind_request, quantity_request
from search.engine.binders.jev_split import RequestTooLarge
from search.engine.config import settings

if TYPE_CHECKING:
    from search.engine.clients.llm import SupportsLLM
    from search.engine.config import SearchSettings
    from search.engine.quantity_read import QuantityOutput

type FallbackCause = Literal["timeout", "429", "5xx", "402", "breaker_open", "too_large", "error"]
"""Why a Jev request gave no answer, as the ``binder_part`` line states it."""


class Ask(Protocol):
    """`jev_client.ask`'s signature."""

    def __call__(
        self,
        state: JSONContent,
        questions: Mapping[str, JevQuestion],
        *,
        settings: SearchSettings,
    ) -> Awaitable[JevResponse]: ...


_REASONS: dict[str, FallbackCause] = {
    "timeout": "timeout",
    "slot_wait": "timeout",
    "request_timeout": "timeout",
    "rate_limited": "429",
    "server": "5xx",
    "payment_required": "402",
    "breaker_open": "breaker_open",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class JevCall:
    """One Jev request as the ``binder_part`` line reports it: its wall time, the questions
    it asked and the chunks they went in, and the tokens Jev reported, None for a request
    that failed. `questions` is None for a call that raised before it built a request."""

    ms: int
    questions: int | None = None
    chunks: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def usage(self) -> dict[str, int]:
        """The call's token usage, tagged as Jev's so it stays out of the Gemini
        counts."""
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "jev_input_tokens": self.input_tokens or 0,
            "jev_output_tokens": self.output_tokens or 0,
        }


class JevFailure(Exception):
    """A Jev request that gave no answer: the breaker was open, the request was too large
    to send, or the call failed or overran its ceiling (`jev_client.JevError`). `reason`
    classes it and `kind` names it (the `JevError` kind, or ``too_large``); `call` is the
    failed request's time and question count."""

    def __init__(self, reason: FallbackCause, kind: str, call: JevCall) -> None:
        super().__init__(f"jev {kind}")
        self.reason: FallbackCause = reason
        self.kind = kind
        self.call = call


class JevBinder:
    """Reads the quantity and elects and binds through Jev, on `ask` (`jev_client.ask`
    unless a test injects another). The LLM client the protocol passes is unused."""

    def __init__(self, *, ask: Ask = ask) -> None:
        self._ask = ask

    @property
    def name(self) -> str:
        return "jev"

    @property
    def model(self) -> str:
        return settings().jev_model

    @property
    def reads_query_text(self) -> bool:
        return True

    async def read_quantity(
        self, request: QuantityRequest, *, llm: SupportsLLM
    ) -> tuple[QuantityOutput, dict | None]:
        output, call = await self.quantity(request)
        return output, call.usage

    async def bind(
        self, request: BindRequest, *, llm: SupportsLLM
    ) -> tuple[BindResult, dict | None]:
        result, call = await self.elect(request)
        return result, call.usage

    async def quantity(self, request: QuantityRequest) -> tuple[QuantityOutput, JevCall]:
        """The quantity read and the Jev request it took.

        Raises:
            JevFailure: The request gave no answer.
        """
        state, questions = quantity_request(
            request.variable,
            request.denominators,
            bind_variable=request.bind_variable,
            query=request.query_text,
        )
        answers, call = await self._request(state, questions)
        return assemble_quantity(answers, request.denominators), call

    async def elect(self, request: BindRequest) -> tuple[BindResult, JevCall]:
        """The bind result over `request`'s menu and the Jev request it took. The form the
        frame states (`jev_assemble.frame_stated_form`) is the one the elections read.

        Raises:
            JevFailure: The request gave no answer.
        """
        menu = jev_menu(request)
        state, questions = bind_request(menu)
        answers, call = await self._request(state, questions)
        result, _confidence = bind_result(
            menu, answers, stated_form=frame_stated_form(request.frame)
        )
        return result, call

    async def _request(
        self, state: dict[str, JSONContent], questions: Mapping[str, JevQuestion]
    ) -> tuple[JevAnswers, JevCall]:
        started = time.perf_counter()

        def failed(reason: FallbackCause, kind: str) -> JevFailure:
            ms = round((time.perf_counter() - started) * 1000)
            return JevFailure(reason, kind, JevCall(ms=ms, questions=len(questions)))

        try:
            response = await self._ask(state, questions, settings=settings())
        except JevError as exc:
            raise failed(_REASONS.get(exc.kind, "error"), exc.kind) from exc
        except RequestTooLarge as exc:
            raise failed("too_large", "too_large") from exc
        call = JevCall(
            ms=round(response.seconds * 1000),
            questions=len(questions),
            chunks=response.chunks,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return JevAnswers(nouls=response.nouls, choices=response.choices), call
