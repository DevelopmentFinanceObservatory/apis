"""The Gemini binder answering, with a second binder run beside it on a sample of calls.

`ShadowBinder` serves every quantity read and bind call from Gemini. On a sampled call
(`Settings.jev_shadow_sample`) it also starts the same request as a detached task before it
awaits Gemini: to Jev, or on the control share of sampled calls
(`Settings.jev_shadow_control`) to Gemini again, so the comparison has Gemini's
disagreement with itself to read Jev's against. The control call skips the LLM memo, so it
reaches the model. A shadow call's tokens stay out of the answer's usage, and a read it
swallows stays out of the served part's degraded reads (`degraded_reads`).

The served path never awaits the shadow task, so the shadow adds no wait to the answer: a
shadow request still running when Gemini answers goes on alone, a Jev request bounded by
its own ceiling (`Settings.jev_ceiling_s`), and its failure is logged, never raised. The
shadow bind call runs the house policies on its own copy of the request's policy step, so
the served call's step counts the served call alone.

Once the shadow task ends, the call's ``binder_part`` line is logged with the Jev request's
time, questions, chunks and tokens, and one ``binder_shadow`` line with a JSON object
carrying what an offline replay reads: the part and pass that join it to the part's
``bind_decision`` line, whether it is the control, the phrase and the menu's shape ids or
the denominators, both readings and whether they agree (on a bind call the shape and ask,
the bindings and the alternates apiece), and the shadow request's time, question and chunk
counts and tokens.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from search.engine.binders.base import BindRequest, BindResult, QuantityRequest
from search.engine.binders.jev import JevFailure
from search.engine.binders.part_log import Call, elapsed_ms, log_binder_part
from search.engine.clients.llm import unrecorded_llm_usage, without_memo
from search.engine.degraded_reads import recording_degraded_reads

if TYPE_CHECKING:
    from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
    from search.engine.binders.base import Binder
    from search.engine.binders.jev import JevBinder, JevCall
    from search.engine.clients.llm import SupportsLLM
    from search.engine.quantity_read import QuantityOutput
    from search.engine.shape import ShapeDraft

logger = logging.getLogger(__name__)

type Mode = Literal["jev", "control"]

# Shadow tasks still running, held so the loop does not collect them mid-flight.
_RUNNING: set[asyncio.Task[Any]] = set()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Spent:
    """What a shadow request took: its time, questions, chunks and tokens, each None where
    the request reports none, and the Jev request itself when Jev answered."""

    ms: int
    questions: int | None = None
    chunks: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    jev: JevCall | None = None

    @staticmethod
    def of_jev(call: JevCall) -> _Spent:
        return _Spent(
            ms=call.ms,
            questions=call.questions,
            chunks=call.chunks,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            jev=call,
        )


class ShadowBinder:
    """`served` (Gemini) for every answer, and beside it on the share `sample` of calls
    `shadow` (Jev), or `served` again on the share `control` of those; both shares are drawn
    per call from `draw`. Its name and model are the served binder's, since every answer
    is."""

    def __init__(
        self,
        *,
        served: Binder,
        shadow: JevBinder,
        sample: float,
        control: float = 0.0,
        draw: Callable[[], float] = random.random,
    ) -> None:
        self.served = served
        self.shadow = shadow
        self.sample = sample
        self.control = control
        self._draw = draw

    @property
    def name(self) -> str:
        return self.served.name

    @property
    def model(self) -> str:
        return self.served.model

    @property
    def reads_query_text(self) -> bool:
        return self.served.reads_query_text

    async def read_quantity(
        self, request: QuantityRequest, *, llm: SupportsLLM
    ) -> tuple[QuantityOutput, dict | None]:
        mode = self._mode()
        shadow = None
        if mode == "jev":
            shadow = _detach(_jev(self.shadow.quantity(request)))
        elif mode == "control":
            shadow = _detach(_control(self.served.read_quantity(request, llm=without_memo(llm))))
        started = time.perf_counter()
        answer = await _served(shadow, self.served.read_quantity(request, llm=llm))
        ms = elapsed_ms(started)
        if shadow is None:
            log_binder_part(call="quantity", binder="shadow", served_by=self.served.name, ms=ms)
            return answer
        record: dict[str, object] = {
            "call": "quantity",
            "part": request.part,
            "pass": request.pool,
            "control": mode == "control",
            "phrase": request.variable,
            "bind_phrase": request.bind_variable,
            "denominators": [ref.ref for ref in request.denominators],
        }
        _log_when_done(
            shadow,
            record,
            served=_quantity_reading(answer[0]),
            reading=_quantity_reading,
            served_by=self.served.name,
            started=started,
        )
        return answer

    async def bind(
        self, request: BindRequest, *, llm: SupportsLLM
    ) -> tuple[BindResult, dict | None]:
        mode = self._mode()
        shadow = None
        if mode is not None:
            own_step = dataclasses.replace(
                request, apply_policies=copy.copy(request.apply_policies)
            )
            shadow = _detach(
                _jev(self.shadow.elect(own_step))
                if mode == "jev"
                else _control(self.served.bind(own_step, llm=without_memo(llm)))
            )
        started = time.perf_counter()
        answer = await _served(shadow, self.served.bind(request, llm=llm))
        ms = elapsed_ms(started)
        if shadow is None:
            log_binder_part(call="bind", binder="shadow", served_by=self.served.name, ms=ms)
            return answer
        shapes = request.shapes
        record: dict[str, object] = {
            "call": "bind",
            "part": request.part,
            "pass": request.pool,
            "control": mode == "control",
            "phrase": request.variable,
            "menu": [shape.shape_id for shape in shapes],
        }
        _log_when_done(
            shadow,
            record,
            served=_bind_reading(answer[0].output, shapes),
            reading=lambda result: _bind_reading(result.output, shapes),
            served_by=self.served.name,
            started=started,
        )
        return answer

    def _mode(self) -> Mode | None:
        """Whether this call is shadowed, and by Jev or by Gemini as the control."""
        if self._draw() >= self.sample:
            return None
        return "control" if self._draw() < self.control else "jev"


async def _jev[R](call: Awaitable[tuple[R, JevCall]]) -> tuple[R, _Spent]:
    answer, jev_call = await call
    return answer, _Spent.of_jev(jev_call)


async def _control[R](call: Awaitable[tuple[R, dict | None]]) -> tuple[R, _Spent]:
    started = time.perf_counter()
    answer, usage = await call
    usage = usage or {}
    spent = _Spent(
        ms=elapsed_ms(started),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )
    return answer, spent


async def _isolated[T](call: Awaitable[T]) -> T:
    """`call` with its model usage and its swallowed reads kept out of the request's
    records, which the task's copy of the request context still points at: a comparison
    call's tokens are not the answer's, and its failed read leaves the served part whole."""
    with unrecorded_llm_usage(), recording_degraded_reads(detached=True):
        return await call


def _detach[T](call: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    task = asyncio.create_task(_isolated(call))
    _RUNNING.add(task)
    task.add_done_callback(_RUNNING.discard)
    return task


async def _served[T](shadow: asyncio.Task[Any] | None, call: Coroutine[Any, Any, T]) -> T:
    """The served answer; a served call that raises cancels the shadow task."""
    try:
        return await call
    except BaseException:
        if shadow is not None:
            shadow.cancel()
        raise


def _log_when_done[R](
    shadow: asyncio.Task[tuple[R, _Spent]],
    record: dict[str, object],
    *,
    served: dict[str, object],
    reading: Callable[[R], dict[str, object]],
    served_by: str,
    started: float,
) -> None:
    """Log the call's ``binder_part`` line and its ``binder_shadow`` line once `shadow`
    ends: `record` with the served reading, the shadow's, whether they agree, and the
    shadow request's cost, or why it gave no answer. A cancelled shadow logs the
    ``binder_part`` line alone. `started` is when both calls began, and the served call's
    time ends when `_log_when_done` is called."""
    call: Call = "bind" if record["call"] == "bind" else "quantity"
    ms = elapsed_ms(started)

    def done(task: asyncio.Task[tuple[R, _Spent]]) -> None:
        if task.cancelled():
            log_binder_part(call=call, binder="shadow", served_by=served_by, ms=ms)
            return
        exc = task.exception()
        shadow_reading: dict[str, object] | None = None
        failure: str | None = None
        if exc is None:
            answer, spent = task.result()
            shadow_reading = reading(answer)
        elif isinstance(exc, JevFailure):
            spent, failure = _Spent.of_jev(exc.call), exc.reason
        else:
            logger.warning("shadow %s call raised", call, exc_info=exc)
            spent, failure = _Spent(ms=elapsed_ms(started)), "error"
        log_binder_part(call=call, binder="shadow", served_by=served_by, ms=ms, jev=spent.jev)
        line = record | {
            "shadow_binder": "gemini" if record["control"] else "jev",
            **_agreement(call, served, shadow_reading),
            "failure": failure,
            "served": served,
            "shadow": shadow_reading,
            "served_ms": ms,
            "shadow_ms": spent.ms,
            "questions": spent.questions,
            "chunks": spent.chunks,
            "input_tokens": spent.input_tokens,
            "output_tokens": spent.output_tokens,
        }
        logger.info("binder_shadow %s", json.dumps(line, separators=(",", ":")))

    shadow.add_done_callback(done)


def _agreement(
    call: Call, served: dict[str, object], shadow: dict[str, object] | None
) -> dict[str, bool | None]:
    """Whether the shadow read what Gemini read, each None when the shadow gave no answer.

    ``agree`` compares the elected shape and the ask on a bind call, the form and base on a
    quantity read. A bind call also states ``agree_bindings``, the same constraining
    bindings, and ``agree_also_plausible``, the same alternates in any order."""
    keys = ("shape", "ask") if call == "bind" else ("form", "denominator")
    agree = None if shadow is None else all(served[key] == shadow[key] for key in keys)
    if call != "bind":
        return {"agree": agree}
    if shadow is None:
        return {"agree": None, "agree_bindings": None, "agree_also_plausible": None}
    return {
        "agree": agree,
        "agree_bindings": served["bindings"] == shadow["bindings"],
        "agree_also_plausible": sorted(cast("list[str]", served["also_plausible"]))
        == sorted(cast("list[str]", shadow["also_plausible"])),
    }


def _quantity_reading(output: QuantityOutput) -> dict[str, object]:
    return {"form": output.stated_form, "denominator": output.denominator_index}


def _bind_reading(output: _UniversalBindOutput, shapes: list[ShapeDraft]) -> dict[str, object]:
    """The output canonically: the elected shape's id, whether it asks, each binding that
    constrains the answer as its property and value ids (``*`` for every value the phrase
    asks for), and the alternates' shape ids. A slot left open has no entry, since
    binders differ in whether they list one."""
    elects = not output.ask and 0 <= output.chosen_shape_index < len(shapes)
    return {
        "shape": shapes[output.chosen_shape_index].shape_id if elects else None,
        "ask": bool(output.ask),
        "bindings": sorted(
            f"{draft.property_dcid}={_bound_values(draft)}"
            for draft in output.bindings
            if draft.value_dcids or draft.evidence == "requested_all"
        ),
        "also_plausible": [
            shapes[i].shape_id for i in output.also_plausible or () if 0 <= i < len(shapes)
        ],
    }


def _bound_values(draft: SlotBindingDraft) -> str:
    return "*" if draft.evidence == "requested_all" else ",".join(draft.value_dcids)
