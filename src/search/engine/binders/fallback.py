"""The Jev binder with the Gemini binder behind it.

`FallbackBinder` serves each quantity read and bind call from its primary, the Jev binder,
and from its fallback, the Gemini binder, when the Jev path gave no answer: a
`jev.JevFailure` (a timeout or overrun ceiling, a 429, a 5xx, a 402, an open breaker, a
request too large to send, any other Jev error), or any other exception, which is a defect
and is logged as one. A bind call it serves from the fallback says so in
`BindResult.fallback_reason`, and every call it serves from the fallback is noted in the
enclosing `recording_fallbacks` record, so the request that holds it can decline to cache
an answer its configured binder did not give. A successful fallback is a normal answer: it
marks nothing degraded.

A Jev request fails before its answer is assembled, so the house policies have not run when
the fallback starts. A bind call whose Jev path raised after they ran re-raises, so each
bind call runs them at most once whichever binder serves it.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from search.engine.binders.base import BindRequest, BindResult, QuantityRequest
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.binders.jev import FallbackCause, JevCall, JevFailure
from search.engine.binders.part_log import Call, elapsed_ms, log_binder_part

if TYPE_CHECKING:
    from search.engine.binders.base import Binder
    from search.engine.binders.jev import JevBinder
    from search.engine.clients.llm import SupportsLLM
    from search.engine.quantity_read import QuantityOutput

logger = logging.getLogger(__name__)


@dataclass
class Fallbacks:
    """The reasons of the calls a request served from a fallback binder, in call order."""

    reasons: list[FallbackCause] = field(default_factory=list)


_RECORD: ContextVar[Fallbacks | None] = ContextVar("search_binder_fallbacks", default=None)


@contextmanager
def recording_fallbacks() -> Iterator[Fallbacks]:
    """Collect, for the body of the `with` block and the tasks it starts, the reason of
    each call a `FallbackBinder` serves from its fallback."""
    record = Fallbacks()
    token = _RECORD.set(record)
    try:
        yield record
    finally:
        _RECORD.reset(token)


def _note_fallback(reason: FallbackCause) -> None:
    record = _RECORD.get()
    if record is not None:
        record.reasons.append(reason)


class FallbackBinder:
    """`primary` (Jev) for every call, and `fallback` (Gemini) for a call the Jev path did
    not answer. Its name and model are the primary's, which the response cache key carries.

    A `JevFailure` falls back under its own cause. Any other exception from the Jev path is
    a defect: it is logged at ERROR with its traceback and falls back under ``error``, so
    the part still answers. A bind call whose Jev path raised after the house policies ran
    on its election re-raises instead, since the fallback's election would run them a
    second time. Cancellation propagates."""

    def __init__(self, *, primary: JevBinder, fallback: Binder) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def name(self) -> str:
        return self.primary.name

    @property
    def model(self) -> str:
        return self.primary.model

    @property
    def reads_query_text(self) -> bool:
        return self.primary.reads_query_text

    async def read_quantity(
        self, request: QuantityRequest, *, llm: SupportsLLM
    ) -> tuple[QuantityOutput, dict | None]:
        started = time.perf_counter()
        try:
            output, call = await self.primary.quantity(request)
        except Exception as exc:
            failure = _failure("quantity", exc, started=started)
            answer = await self.fallback.read_quantity(request, llm=llm)
            log_binder_part(
                call="quantity",
                binder=self.name,
                served_by=self.fallback.name,
                ms=elapsed_ms(started),
                failure=failure,
            )
            return answer
        log_binder_part(
            call="quantity", binder=self.name, served_by=self.name, ms=elapsed_ms(started), jev=call
        )
        return output, call.usage

    async def bind(
        self, request: BindRequest, *, llm: SupportsLLM
    ) -> tuple[BindResult, dict | None]:
        started = time.perf_counter()
        policies_ran = False

        def apply_policies(
            pool: PolicyPool,
            *,
            ratings: Mapping[int, float] | None = None,
            holds: Mapping[int, bool] | None = None,
        ) -> PolicyChoice:
            nonlocal policies_ran
            policies_ran = True
            return request.apply_policies(pool, ratings=ratings, holds=holds)

        try:
            result, call = await self.primary.elect(
                dataclasses.replace(request, apply_policies=apply_policies)
            )
        except Exception as exc:
            if policies_ran:
                raise
            failure = _failure("bind", exc, started=started)
            result, usage = await self.fallback.bind(request, llm=llm)
            log_binder_part(
                call="bind",
                binder=self.name,
                served_by=self.fallback.name,
                ms=elapsed_ms(started),
                failure=failure,
            )
            return dataclasses.replace(result, fallback_reason=failure.reason), usage
        log_binder_part(
            call="bind", binder=self.name, served_by=self.name, ms=elapsed_ms(started), jev=call
        )
        return result, call.usage


def _failure(call: Call, exc: Exception, *, started: float) -> JevFailure:
    """`exc` as the failure a fallback records: itself when it is a `JevFailure`, else a
    defect logged at ERROR and recorded under ``error``."""
    if not isinstance(exc, JevFailure):
        logger.error("jev %s call raised; falling back to Gemini", call, exc_info=exc)
        exc = JevFailure("error", type(exc).__name__, JevCall(ms=elapsed_ms(started)))
    _note_fallback(exc.reason)
    return exc
