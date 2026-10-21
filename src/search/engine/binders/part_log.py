"""The ``binder_part`` line: one per quantity read and per bind call, naming the binder the
request was routed to, the one that served the call, why it fell back when it did, and the
Jev request's time, questions, chunks and tokens when one was sent. A Jev bind also states
the questions of its span round (``span_questions``, ``-`` when none ran) and the values the
span rule left unbound (``inferred``, ``property=value`` pairs joined by commas, ``-`` for
none). Fallback share, 402 and 429 counts and Jev latency are log-based metrics on this
line.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

from search.engine.binders.base import BindRequest, BindResult, QuantityRequest

if TYPE_CHECKING:
    from search.engine.binders.base import Binder
    from search.engine.binders.jev import JevCall, JevFailure
    from search.engine.clients.llm import SupportsLLM
    from search.engine.quantity_read import QuantityOutput

logger = logging.getLogger(__name__)

type Call = Literal["quantity", "bind"]


def elapsed_ms(started: float) -> int:
    """Milliseconds since `started` (`time.perf_counter`)."""
    return round((time.perf_counter() - started) * 1000)


def log_binder_part(
    *,
    call: Call,
    binder: str,
    served_by: str,
    ms: int,
    jev: JevCall | None = None,
    failure: JevFailure | None = None,
) -> None:
    """Log one ``binder_part`` line. `binder` is the binder the request was routed to and
    `served_by` the one whose answer the part reads; `jev` is the Jev request that answered,
    and `failure` the one that did not."""
    sent = failure.call if failure is not None else jev
    counts = (
        ("-",) * 6
        if sent is None
        else tuple(
            "-" if n is None else str(n)
            for n in (
                sent.ms,
                sent.questions,
                sent.chunks,
                sent.input_tokens,
                sent.output_tokens,
                sent.span_questions,
            )
        )
    )
    inferred = ",".join(sent.inferred) if sent is not None and sent.inferred else "-"
    logger.info(
        "binder_part call=%s binder=%s served_by=%s fallback=%s jev_error=%s ms=%d jev_ms=%s "
        "questions=%s chunks=%s jev_input_tokens=%s jev_output_tokens=%s span_questions=%s "
        "inferred=%s",
        call,
        binder,
        served_by,
        "-" if failure is None else failure.reason,
        "-" if failure is None else failure.kind,
        ms,
        *counts,
        inferred,
    )


class DirectBinder:
    """`binder` serving every call itself, with its ``binder_part`` line."""

    def __init__(self, binder: Binder) -> None:
        self.binder = binder

    @property
    def name(self) -> str:
        return self.binder.name

    @property
    def model(self) -> str:
        return self.binder.model

    @property
    def reads_query_text(self) -> bool:
        return self.binder.reads_query_text

    async def read_quantity(
        self, request: QuantityRequest, *, llm: SupportsLLM
    ) -> tuple[QuantityOutput, dict | None]:
        started = time.perf_counter()
        answer = await self.binder.read_quantity(request, llm=llm)
        log_binder_part(
            call="quantity", binder=self.name, served_by=self.name, ms=elapsed_ms(started)
        )
        return answer

    async def bind(
        self, request: BindRequest, *, llm: SupportsLLM
    ) -> tuple[BindResult, dict | None]:
        started = time.perf_counter()
        answer = await self.binder.bind(request, llm=llm)
        log_binder_part(call="bind", binder=self.name, served_by=self.name, ms=elapsed_ms(started))
        return answer
