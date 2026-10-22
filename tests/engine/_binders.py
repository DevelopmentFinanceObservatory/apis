"""Binder doubles: a scripted elected bind reply that rates its pick, the Gemini binder
with its menus recorded, one that elects without a model call, one that answers with the
reply its model double scripts, one under a chosen name and model, any binder rating the
words the coverage rule leaves unread as a test scripts, a house-policy step that keeps the
binder's pick, a recorded-fixture LLM whose bind call answers with a fixed reply, and Jev's
`ask` answering without the network."""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping, Sequence

from search.engine.bind import ShapeRating, SlotBindingDraft, _UniversalBindOutput, bind_universal
from search.engine.binders import (
    Binder,
    BindRequest,
    BindResult,
    GeminiBinder,
    QuantityRequest,
    StatedWordsRequest,
)
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from search.engine.binders.jev_client import JevResponse, JevUsage
from tests.fixtures import FakeLLM

# The rating a scripted reply gives each shape it nominates: the lowest a nomination stands
# at beside a pick rated 1 (`gemini.nominated_near_best`), so the reply still prefers its
# pick.
NOMINEE_RATING = 0.9


def nominee_ratings(
    also_plausible: Sequence[int] | None, *, rated: Collection[int] = ()
) -> list[ShapeRating]:
    """A rating at `NOMINEE_RATING` for each shape of `also_plausible` outside `rated`, the
    shapes a scripted reply already rates, so each of its nominations stands."""
    return [
        ShapeRating(index=index, rating=NOMINEE_RATING)
        for index in dict.fromkeys(also_plausible or ())
        if index not in rated
    ]


def elected_reply(
    chosen_shape_index: int,
    *,
    bindings: Sequence[SlotBindingDraft] = (),
    also_plausible: Sequence[int] | None = None,
    unplaced: Sequence[str] = (),
    rating: float | None = 1.0,
) -> _UniversalBindOutput:
    """A bind reply electing `chosen_shape_index` that rates its pick at `rating` and each
    shape it nominates in `also_plausible` at `NOMINEE_RATING` in `measuring`, as the bind
    prompt asks of a reply that elects. The Gemini binder reads a pick the reply leaves
    unrated as a decline, so a test pinning that decline passes `rating=None`."""
    return _UniversalBindOutput(
        chosen_shape_index=chosen_shape_index,
        bindings=list(bindings),
        also_plausible=None if also_plausible is None else list(also_plausible),
        unplaced=list(unplaced),
        measuring=[
            *([] if rating is None else [ShapeRating(index=chosen_shape_index, rating=rating)]),
            *nominee_ratings(also_plausible, rated=() if rating is None else (chosen_shape_index,)),
        ],
    )


def keep_the_pick(
    pool: PolicyPool,
    *,
    ratings: Mapping[int, float] | None = None,
    holds: Mapping[int, bool] | None = None,
) -> PolicyChoice:
    """A house-policy step that elects the binder's pick, the pool's first index."""
    return PolicyChoice(survivor=pool.measuring[0], fired=(), dropped=frozenset(), facts=())


class MenuRecordingBinder(GeminiBinder):
    """The Gemini binder, recording the StatVars of each menu shape of every bind call."""

    def __init__(self) -> None:
        self.menus: list[list[tuple[str, ...]]] = []

    async def bind(self, request: BindRequest, *, llm) -> tuple[BindResult, dict | None]:
        self.menus.append([tuple(shape.sv_arc_facts or ()) for shape in request.shapes])
        return await super().bind(request, llm=llm)


class FixedBinder(GeminiBinder):
    """Answers every bind call with `output` and `usage`, without a model call, and keeps
    each request it was sent. An output that elects a shape goes through the house policies
    once, over a pool of that shape alone, so they keep it. The quantity read stays the
    Gemini binder's."""

    def __init__(
        self, output: _UniversalBindOutput, *, usage: dict | None = None, name: str = "fixed"
    ) -> None:
        self._output = output
        self._usage = usage
        self._name = name
        self.requests: list[BindRequest] = []

    @property
    def name(self) -> str:
        return self._name

    async def bind(self, request: BindRequest, *, llm) -> tuple[BindResult, dict | None]:
        self.requests.append(request)
        output = self._output
        if output.ask or not 0 <= output.chosen_shape_index < len(request.shapes):
            return BindResult(output=output, binder=self.name), self._usage
        pool = PolicyPool(measuring=(output.chosen_shape_index,))
        choice = request.apply_policies(pool)
        result = BindResult(output=output, binder=self.name, policy_pool=pool, policy_choice=choice)
        return result, self._usage


class ReplyBinder(GeminiBinder):
    """Binds with `bind.bind_universal` on the injected LLM, as the Gemini binder does, and
    answers with the reply as it comes: an election goes through the house policies once,
    over a pool of its pick alone, so they keep it, and its nominations stand. A test of
    machinery downstream of the bind result hands it the reply its model double scripts."""

    @property
    def name(self) -> str:
        return "reply"

    async def bind(self, request: BindRequest, *, llm) -> tuple[BindResult, dict | None]:
        output, usage = await bind_universal(
            request.variable,
            request.shapes,
            admitted_qualifiers=request.admitted_qualifiers,
            admitted_ties=request.admitted_ties,
            withheld_measure_axes=request.withheld_measure_axes,
            dropped_qualifier_phrases=request.dropped_qualifier_phrases,
            candidate_menus=request.candidate_menus,
            frame=request.frame,
            refused=request.refused,
            availability=request.availability,
            llm=llm,
        )
        if output.ask or not 0 <= output.chosen_shape_index < len(request.shapes):
            return BindResult(output=output, binder=self.name), usage
        pool = PolicyPool(measuring=(output.chosen_shape_index,))
        choice = request.apply_policies(pool)
        result = BindResult(output=output, binder=self.name, policy_pool=pool, policy_choice=choice)
        return result, usage


class FixedBindReplyLLM:
    """The recorded fixtures (`FakeLLM`) for every call but the bind call, which answers
    with `reply` and keeps its prompt. A test that pins what the engine does with one bind
    reply holds that reply fixed while the rest of the pipeline replays its recordings."""

    def __init__(self, reply: _UniversalBindOutput) -> None:
        self._reply = reply
        self._delegate = FakeLLM()
        self.bind_prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "_UniversalBindOutput":
            self.bind_prompts.append(prompt)
            return self._reply, None
        return await self._delegate.generate_structured(prompt=prompt, system=system, schema=schema)


class KeyedBindReplyLLM:
    """`delegate`, the recorded fixtures (`FakeLLM`) by default, for every call but a bind
    call whose variable phrase is a key of `replies`, which answers with that key's reply.
    A test that needs one part's reply scripted keeps the recordings for the rest."""

    def __init__(
        self,
        replies: Mapping[str, _UniversalBindOutput],
        *,
        delegate: FakeLLM | None = None,
    ) -> None:
        self._replies = replies
        self._delegate = delegate if delegate is not None else FakeLLM()

    async def generate_structured(self, *, prompt, system, schema):
        variable = prompt.partition("\n")[0].removeprefix("Variable phrase: ")
        if schema.__name__ == "_UniversalBindOutput" and variable in self._replies:
            return self._replies[variable], None
        return await self._delegate.generate_structured(prompt=prompt, system=system, schema=schema)


# "birth rate" over the offline Ethiopia menu: the crude birth rate elected with
# `medicalStatus` open, and the fertility rate nominated near the best, so the part answers
# candidates.
BIRTH_RATE_CANDIDATES = {
    "birth rate": elected_reply(
        0,
        bindings=[SlotBindingDraft(axis="how", property_dcid="medicalStatus", kind="unbound")],
        also_plausible=[1],
    )
}


class KeyedBinder(GeminiBinder):
    """The Gemini binder under `name` and `model`, stating whether it reads the query
    text, as the response cache keys a binder."""

    def __init__(
        self, *, name: str = "gemini", model: str = "model-a", reads_query_text: bool = False
    ) -> None:
        self._name = name
        self._model = model
        self._reads_query_text = reads_query_text

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def reads_query_text(self) -> bool:
        return self._reads_query_text


class RatingBinder:
    """`binder` for every call but the rating of the words the coverage rule leaves unread
    (`Binder.rate_words_stated`), which answers each word `stated` holds with its
    probability, so a word below `SearchSettings.stated_word_bar` is unread, and rates
    every other word 1.0, stated. Each rating request is kept."""

    def __init__(self, binder: Binder, stated: Mapping[str, float] | None = None) -> None:
        self.binder = binder
        self.stated = stated or {}
        self.requests: list[StatedWordsRequest] = []

    @property
    def name(self) -> str:
        return self.binder.name

    @property
    def model(self) -> str:
        return self.binder.model

    @property
    def reads_query_text(self) -> bool:
        return self.binder.reads_query_text

    async def read_quantity(self, request: QuantityRequest, *, llm):
        return await self.binder.read_quantity(request, llm=llm)

    async def bind(self, request: BindRequest, *, llm) -> tuple[BindResult, dict | None]:
        return await self.binder.bind(request, llm=llm)

    async def rate_words_stated(
        self, request: StatedWordsRequest, *, llm
    ) -> tuple[Mapping[str, float], dict | None]:
        self.requests.append(request)
        return {word: self.stated.get(word, 1.0) for word in request.words}, None


class FakeAsk:
    """`jev_client.ask` answering from `answers` (`answer_all` when None), or raising
    `fail`, and recording each request. A `gate` holds every answer until it is set. A
    noul asking whether menu shape i's series states word w (``unread|i|w``) answers
    `stated`'s probability for w where it holds one, else as `answers` do: yes at 0.9
    under `answer_all`, stated."""

    def __init__(
        self,
        answers: Mapping[str, dict] | None = None,
        *,
        fail: Exception | None = None,
        gate: asyncio.Event | None = None,
        stated: Mapping[str, float] | None = None,
    ) -> None:
        self.answers = answers
        self.fail = fail
        self.gate = gate
        self.stated = stated or {}
        self.requests: list[tuple[object, dict]] = []

    async def __call__(self, state, questions, *, settings) -> JevResponse:
        self.requests.append((state, dict(questions)))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        stored = self.answers if self.answers is not None else answer_all(questions)
        asked = {qid: stored[qid] for qid in questions if qid in stored}
        for qid in questions:
            if qid.startswith("unread|") and (word := qid.rpartition("|")[2]) in self.stated:
                asked[qid] = {"type": "noul", "noul": self.stated[word]}
        return JevResponse(
            nouls={k: float(v["noul"]) for k, v in asked.items() if v["type"] == "noul"},
            choices={
                k: {o: float(p) for o, p in v["probabilities"].items()}
                for k, v in asked.items()
                if v["type"] == "choice"
            },
            model="jev-1.13.0",
            usage=JevUsage(input_tokens=120, output_tokens=4),
            seconds=0.2,
            chunks=1,
        )


def answer_all(questions: Mapping[str, dict]) -> dict[str, dict]:
    """Yes at 0.9 to every noul, and every choice's first option at 0.9."""
    answers: dict[str, dict] = {}
    for qid, question in questions.items():
        if question["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.9}
        else:
            first, *rest = question["criteria"]
            probabilities = {first: 0.9} | dict.fromkeys(rest, 0.1 / max(len(rest), 1))
            answers[qid] = {"type": "choice", "probabilities": probabilities}
    return answers
