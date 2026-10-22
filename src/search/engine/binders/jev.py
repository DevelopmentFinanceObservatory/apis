"""The Jev binder: the quantity read and the bind call as two Jev requests.

The quantity read sends `jev_request.quantity_request` and reads it with
`jev_assemble.assemble_quantity`, in place where the Gemini read runs, over the admissible
shapes' denominators. The bind call sends `jev_request.bind_request` over the menu, then,
when its answers call for one (`jev_assemble.span_round`), a second request asking which
words name each contending value, and assembles both rounds' answers with
`jev_assemble.bind_result`, which runs the request's house-policy step once on an answer
that elects a shape. The words the bind rated as unread on the elected shape are asked
again in a request of their own when the final bindings restrict the shape to a value
(`JevBinder.restated`). Every failure of a Jev request, in either round, leaves as one
`JevFailure`, raised before any answer is assembled, and so does a span rule that passes
its search budget, raised before the house policies run, so a binder that falls back on it
runs the house policies at most once per call.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol

from typesafe_sdk import JSONContent

from code_index.shared import get_index
from search.engine.binders.base import (
    BindRequest,
    BindResult,
    QuantityRequest,
    StatedWordsRequest,
)
from search.engine.binders.jev_assemble import (
    JevAnswers,
    Population,
    SpanSearchExhausted,
    assemble_quantity,
    bind_result,
    frame_stated_form,
    span_round,
)
from search.engine.binders.jev_client import JevError, JevQuestion, JevResponse, ask
from search.engine.binders.jev_menu import jev_menu, restricting
from search.engine.binders.jev_request import (
    bind_request,
    quantity_request,
    restated_request,
    unread_id,
)
from search.engine.binders.jev_split import RequestTooLarge
from search.engine.config import settings
from search.engine.errors import LLMInfraError
from search.engine.retrieval.bound_values import (
    held_family_arcs,
    held_family_exact,
    index_member_rows,
    shape_query,
)
from search.engine.retrieval.facts import read_constraints
from search.engine.shape import ShapeDraft

if TYPE_CHECKING:
    from search.engine.clients.llm import SupportsLLM
    from search.engine.config import SearchSettings
    from search.engine.quantity_read import QuantityOutput

type FallbackCause = Literal[
    "timeout", "429", "5xx", "402", "breaker_open", "too_large", "span_budget", "error"
]
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
    "queued_timeout": "timeout",
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
    that failed. `questions` is None for a call that raised before it built a request.

    A bind's two rounds report as one call (`plus`): `span_questions` counts the span
    round's questions, None when none ran, and `inferred` names each value the span rule
    left unbound as ``property=value`` (`jev_assemble.Confidence.inferred`)."""

    ms: int
    questions: int | None = None
    chunks: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    span_questions: int | None = None
    inferred: tuple[str, ...] = ()

    def plus(self, other: JevCall) -> JevCall:
        """This call and `other`, a bind's span round, as one call. Times, questions, chunks
        and tokens are summed, a field either call lacks is taken from the other, and
        `other`'s questions are counted again as `span_questions`."""

        def total(a: int | None, b: int | None) -> int | None:
            return b if a is None else a if b is None else a + b

        return JevCall(
            ms=self.ms + other.ms,
            questions=total(self.questions, other.questions),
            chunks=total(self.chunks, other.chunks),
            input_tokens=total(self.input_tokens, other.input_tokens),
            output_tokens=total(self.output_tokens, other.output_tokens),
            span_questions=other.questions,
        )

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
    to send, or the call failed or overran its ceiling (`jev_client.JevError`), or a bind
    whose answers the span rule could not settle within its search budget
    (`jev_assemble.SpanSearchExhausted`). `reason` classes it and `kind` names it (the
    `JevError` kind, ``too_large`` or ``span_budget``); `call` is the failed request's time
    and question count."""

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

    async def rate_words_stated(
        self, request: StatedWordsRequest, *, llm: SupportsLLM
    ) -> tuple[Mapping[str, float], dict | None]:
        """The ratings `restated` gives the elected shape's words, and the usage of its
        request, None when it sent none.

        Raises:
            LLMInfraError: The request gave no answer.
        """
        try:
            ratings, call = await self.restated(request)
        except JevFailure as exc:
            raise LLMInfraError(f"jev stated words read: {exc.kind}") from exc
        return ratings, None if call is None else call.usage

    async def restated(
        self, request: StatedWordsRequest
    ) -> tuple[dict[str, float], JevCall | None]:
        """Each word of `request` the bind's answers rate on the elected shape
        (`stated_words`), with the words rated below `SearchSettings.stated_word_bar` rated
        again in one request (`jev_request.restated_request`) when the final bindings
        restrict the shape to a value (`jev_menu.restricting`), and the request, None when
        none was sent.

        The bind asked before any value was bound, so it quoted the shape as parts
        selectable by its slots, and its answer is evidence only about the series as a
        whole: a completion rate rates "girls" as unread until Sex=Female is bound. The
        request's answer replaces it, and a word the request leaves unanswered is unrated.
        A word at or above the bar, and a shape with no bound value, keep the bind's
        rating and send no request.

        Raises:
            JevFailure: The request gave no answer.
        """
        ratings = self.stated_words(request)
        bar = settings().stated_word_bar
        unread = tuple(word for word in request.words if ratings.get(word, bar) < bar)
        index = request.shape_index
        if not unread or index is None or not restricting(request.bindings):
            return ratings, None
        state, questions = restated_request(
            request.variable, request.shape, index, unread, bindings=request.bindings
        )
        answers, call = await self._request(state, questions)
        for k, word in enumerate(unread):
            del ratings[word]
            if (qid := unread_id(index, k)) in answers.nouls:
                ratings[word] = float(answers.nouls[qid])
        return ratings, call

    def stated_words(self, request: StatedWordsRequest) -> dict[str, float]:
        """Each word of `request` the bind's answers rate on the elected shape
        (`BindResult.stated_words`); none on a shape outside the menu."""
        decision, index = request.decision, request.shape_index
        if decision is None or index is None:
            return {}
        return {
            word: decision.stated_words[(index, word)]
            for word in request.words
            if (index, word) in decision.stated_words
        }

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
        """The bind result over `request`'s menu and the Jev requests it took, as one call.
        The form the frame states (`jev_assemble.frame_stated_form`) is the one the
        elections read.

        The span round (`jev_assemble.span_round`) asks over the first request's state, and
        the member populations of the shapes it contends (`member_population`) are read off
        the event loop.

        Raises:
            JevFailure: Either request gave no answer, or the span rule passed its search
                budget (``span_budget``), which it does before the house policies run. A
                failed span round, and a passed budget, carry both requests' time and
                questions.
        """
        menu = jev_menu(request)
        form = frame_stated_form(request.frame)
        state, questions = bind_request(menu)
        answers, call = await self._request(state, questions)
        spans = span_round(menu, answers, stated_form=form)
        populations: dict[int, Population] = {}
        if spans.questions:
            try:
                more, second = await self._request(state, spans.questions)
            except JevFailure as exc:
                raise JevFailure(exc.reason, exc.kind, call.plus(exc.call)) from exc
            answers, call = answers.merged(more), call.plus(second)
            populations = await asyncio.to_thread(member_populations, request.shapes, spans.shapes)
        try:
            result, confidence = bind_result(
                menu, answers, stated_form=form, populations=populations
            )
        except SpanSearchExhausted as exc:
            raise JevFailure("span_budget", "span_budget", call) from exc
        inferred = tuple(
            f"{prop.partition(':')[2]}={value}"
            for prop, values in confidence.inferred.items()
            for value in values
        )
        return result, replace(call, inferred=inferred)

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


def member_populations(
    shapes: Sequence[ShapeDraft], indices: Sequence[int]
) -> dict[int, Population]:
    """The complete member population of each shape of `shapes` at `indices` that has one
    (`member_population`)."""
    return {
        i: population for i in indices if (population := member_population(shapes[i])) is not None
    }


def member_population(shape: ShapeDraft) -> Population | None:
    """The constraints of every member a binding of `shape`'s slots could elect, or None
    when no source lists them all. Reads the code index and the catalogue's tables, and no
    graph.

    The population is the whole family, before any dataset or role class is elected, since
    a later source facet or entity role can elect members outside the subset an unbound
    query elects. It is one of:

    - the shape's own members, when they are the exact elected set
      (`ShapeDraft.sv_arc_facts_complete`);
    - the listed rows of its five-tuple (`CodeIndex.listed_by_five_tuple`) with the index
      rows of its family, which the export verifies are every StatVar of the five-tuple, so
      the graph walk elects among them alone (`code_index.members.listed_group`), as for
      World Bank IDS;
    - every row of the family the index carries for it, whatever dataset publishes the row
      (`bound_values.index_member_rows`);
    - every member of the family the catalogue holds whole for it
      (`bound_values.held_family_exact`), of every role class
      (`bound_values.held_family_arcs`).

    A recalled sample, and a held family the walk may extend, are no population. A
    population spanning several datasets or role classes is a superset of what any one
    binding elects: realisability over it can keep a value no elected subset holds, which
    the engine then refuses on loudly, and a value narrows nothing over it less often.
    """
    if shape.sv_arc_facts_complete:
        return tuple(read_constraints(arcs) for arcs in (shape.sv_arc_facts or {}).values())
    index = get_index()
    rows = index_member_rows(shape, index)
    if listed := index.listed_by_five_tuple.get(shape_query(shape, (), ()).five_tuple):
        rows = tuple({row.dcid: row for row in (*rows, *listed)}.values())
    if rows:
        return tuple(dict(row.constraint_properties) for row in rows)
    if held_family_exact(shape):
        return tuple(read_constraints(arcs) for arcs in held_family_arcs(shape).values())
    return None
