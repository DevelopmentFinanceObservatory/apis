"""Binder doubles: a scripted elected bind reply that rates its pick, the Gemini binder
with its menus recorded, one that elects without a model call, a house-policy step that
keeps the binder's pick, and a recorded-fixture LLM whose bind call answers with a fixed
reply."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from search.engine.bind import ShapeRating, SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import BindRequest, BindResult, GeminiBinder
from search.engine.binders.house_policy import PolicyChoice, PolicyPool
from tests.fixtures import FakeLLM


def elected_reply(
    chosen_shape_index: int,
    *,
    bindings: Sequence[SlotBindingDraft] = (),
    also_plausible: Sequence[int] | None = None,
    unplaced: Sequence[str] = (),
    rating: float | None = 1.0,
) -> _UniversalBindOutput:
    """A bind reply electing `chosen_shape_index` that rates its pick at `rating` in
    `measuring`, as the bind prompt asks of a reply that elects. The Gemini binder reads a
    pick the reply leaves unrated as a decline, so a test pinning that decline passes
    `rating=None`."""
    return _UniversalBindOutput(
        chosen_shape_index=chosen_shape_index,
        bindings=list(bindings),
        also_plausible=None if also_plausible is None else list(also_plausible),
        unplaced=list(unplaced),
        measuring=[] if rating is None else [ShapeRating(index=chosen_shape_index, rating=rating)],
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
