"""The Jev binder, the fallback to Gemini, the shadow binder and the routing between them,
over an injected `ask` that never reaches the network."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import random
import string

import pytest

from search.engine import config
from search.engine.bind import _UniversalBindOutput
from search.engine.binders import (
    BindRequest,
    BindResult,
    DirectBinder,
    FallbackBinder,
    GeminiBinder,
    JevBinder,
    QuantityRequest,
    ShadowBinder,
    configured_binder,
    recording_fallbacks,
    routes_to_jev,
)
from search.engine.binders import jev as jev_module
from search.engine.binders.house_policy import PolicyPool
from search.engine.binders.jev import JevCall
from search.engine.binders.jev_assemble import (
    assemble_quantity,
    bind_result,
    frame_stated_form,
)
from search.engine.binders.jev_client import JevError
from search.engine.binders.jev_menu import jev_menu
from search.engine.binders.jev_request import bind_request, quantity_request
from search.engine.binders.jev_split import RequestTooLarge
from search.engine.clients import llm as llm_client
from search.engine.degraded_reads import note_degraded_read, recording_degraded_reads
from search.engine.envelope import _sum_usage
from search.engine.errors import LLMInfraError
from search.engine.quantity_read import (
    _QUANTITY_SYSTEM_PROMPT,
    QuantityOutput,
    build_quantity_prompt,
)
from search.models import GraphRef
from tests.engine._binders import FakeAsk, FixedBinder, keep_the_pick
from tests.engine._jev_golden import golden_paths, load_item

_GOLDEN = load_item(golden_paths()[0])
_DENOMINATORS = [GraphRef(ref="Amount_EconomicActivity_GrossDomesticProduction", label="GDP")]
_QUANTITY = QuantityRequest(
    variable="debt to GDP",
    denominators=_DENOMINATORS,
    bind_variable="debt to GDP Kenya",
    query_text="debt to GDP in Kenya since 2010",
)
_GEMINI_QUANTITY = QuantityOutput(stated_form="rate")
_GEMINI_USAGE = {"input_tokens": 7, "output_tokens": 3, "cached_tokens": 0}


class _GeminiDouble(FixedBinder):
    """Gemini answering `_GEMINI_QUANTITY` and electing `output`, counting its calls."""

    def __init__(self, output: _UniversalBindOutput, *, delay: float = 0.0) -> None:
        super().__init__(output, usage=_GEMINI_USAGE, name="gemini")
        self.quantity_calls = 0
        self.delay = delay

    async def read_quantity(self, request, *, llm):
        self.quantity_calls += 1
        await asyncio.sleep(self.delay)
        return _GEMINI_QUANTITY, _GEMINI_USAGE

    async def bind(self, request, *, llm):
        await asyncio.sleep(self.delay)
        return await super().bind(request, llm=llm)


class _CountingStep:
    """A house-policy step keeping the binder's pick, counting its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, pool, *, ratings=None, holds=None):
        self.calls += 1
        return keep_the_pick(pool)


def _golden_request(step: _CountingStep) -> BindRequest:
    request = _GOLDEN.request
    return BindRequest(
        variable=request.variable,
        shapes=request.shapes,
        admitted_qualifiers=request.admitted_qualifiers,
        admitted_ties=request.admitted_ties,
        withheld_measure_axes=request.withheld_measure_axes,
        dropped_qualifier_phrases=request.dropped_qualifier_phrases,
        candidate_menus=request.candidate_menus,
        frame=request.frame,
        refused=request.refused,
        availability=request.availability,
        query_text=request.query_text,
        apply_policies=step,
    )


def _with_settings(monkeypatch: pytest.MonkeyPatch, **update: object) -> None:
    monkeypatch.setattr(config, "_SETTINGS", config.settings().model_copy(update=update))


def _binder_parts(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("binder_part")]


# -- routing -----------------------------------------------------------------------------


def test_a_query_keeps_its_binder_across_spellings_the_normalisation_folds():
    picked = [q for q in (f"aid to Kenya {n}" for n in range(200)) if routes_to_jev(q, 30)]

    assert picked
    for query in picked:
        assert routes_to_jev(f"  {query.replace(' ', '   ')}\t", 30)


@pytest.mark.parametrize("percent", [5, 25, 50])
def test_the_share_routed_to_jev_matches_the_percentage(percent: int):
    rng = random.Random(7)
    queries = ["".join(rng.choices(string.ascii_lowercase + " ", k=24)) for _ in range(20_000)]

    share = sum(routes_to_jev(q, percent) for q in queries) / len(queries)

    assert abs(share - percent / 100) < 0.01


def test_no_share_routes_nothing_and_a_full_share_everything():
    assert not routes_to_jev("gdp kenya", 0)
    assert routes_to_jev("gdp kenya", 100)
    assert routes_to_jev(None, 100)
    assert not routes_to_jev(None, 50)


@pytest.mark.parametrize(
    ("setting", "percent", "expected"),
    [
        ("gemini", 100, DirectBinder),
        ("jev", 100, FallbackBinder),
        ("jev", 0, DirectBinder),
        ("shadow", 0, ShadowBinder),
    ],
)
def test_the_setting_routes_to_its_binder(monkeypatch, setting, percent, expected):
    _with_settings(monkeypatch, binder=setting, jev_traffic_pct=percent)

    binder = configured_binder("gdp kenya")

    assert type(binder) is expected
    assert (binder.name, binder.model) == (
        ("jev", "jev-1.13.0")
        if expected is FallbackBinder
        else ("gemini", config.settings().engine_model)
    )


def test_the_jev_setting_alone_routes_every_query_to_jev(monkeypatch):
    monkeypatch.setenv("SEARCH_BINDER", "jev")
    monkeypatch.delenv("SEARCH_JEV_TRAFFIC_PCT", raising=False)
    monkeypatch.setattr(config, "_SETTINGS", config.SearchSettings())

    routed = {type(configured_binder(f"aid to Kenya {n}")) for n in range(200)}

    assert routed == {FallbackBinder}
    assert type(configured_binder(None)) is FallbackBinder


# -- the Jev binder ----------------------------------------------------------------------


def test_the_jev_bind_call_sends_the_bind_request_and_assembles_its_answers():
    """The bind call asks `bind_request` over the menu, and its result is `bind_result` on
    the answers with the frame's stated form; the house policies run once."""
    step = _CountingStep()
    request = _golden_request(step)
    ask = FakeAsk(_GOLDEN.stored)

    result, usage = asyncio.run(JevBinder(ask=ask).bind(request, llm=None))

    ((state, questions),) = ask.requests
    assert (state, questions) == bind_request(jev_menu(request))
    expected, _conf = bind_result(
        jev_menu(_golden_request(_CountingStep())),
        _GOLDEN.answers,
        stated_form=frame_stated_form(request.frame),
    )
    assert result.output == expected.output
    assert result.binder == "jev"
    assert step.calls == (0 if result.output.ask else 1)
    assert usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "jev_input_tokens": 120,
        "jev_output_tokens": 4,
    }


def test_the_jev_quantity_read_is_its_own_request():
    ask = FakeAsk(_GOLDEN.stored)
    request = QuantityRequest(
        variable=_GOLDEN.quantity_variable,
        denominators=_GOLDEN.denominators,
        bind_variable=_GOLDEN.request.variable,
        query_text=_GOLDEN.request.query_text,
    )

    output, _usage = asyncio.run(JevBinder(ask=ask).read_quantity(request, llm=None))

    ((state, questions),) = ask.requests
    assert (state, questions) == quantity_request(
        request.variable,
        request.denominators,
        bind_variable=request.bind_variable,
        query=request.query_text,
    )
    assert output == assemble_quantity(_GOLDEN.answers, _GOLDEN.denominators)


# -- the fallback ------------------------------------------------------------------------

_ELECTS_FIRST = _UniversalBindOutput(chosen_shape_index=0)


@pytest.mark.parametrize(
    ("fail", "reason"),
    [
        (JevError(None, "timeout"), "timeout"),
        (JevError(408, "request_timeout"), "timeout"),
        (JevError(429, "rate_limited"), "429"),
        (JevError(503, "server"), "5xx"),
        (JevError(402, "payment_required"), "402"),
        (JevError(None, "breaker_open"), "breaker_open"),
        (RequestTooLarge("a Jev state exceeds the request budget"), "too_large"),
        (JevError(401, "auth"), "error"),
        (JevError(None, "unanswered"), "error"),
        (RuntimeError("a defect on the Jev path"), "error"),
        (ValueError("a defect that is not an oversized request"), "error"),
    ],
)
def test_a_failed_jev_call_falls_back_to_gemini(caplog, fail, reason):
    """Each failure serves both calls from Gemini, records why on the bind result and in
    the request's fallback record, logs it on the ``binder_part`` line, and runs the house
    policies once."""
    gemini = _GeminiDouble(_ELECTS_FIRST)
    binder = FallbackBinder(primary=JevBinder(ask=FakeAsk(fail=fail)), fallback=gemini)
    step = _CountingStep()
    caplog.set_level(logging.INFO, logger="search.engine.binders.part_log")

    async def run() -> tuple[tuple, tuple]:
        quantity = await binder.read_quantity(_QUANTITY, llm=None)
        bound = await binder.bind(_golden_request(step), llm=None)
        return quantity, bound

    with recording_fallbacks() as record:
        quantity, (result, usage) = asyncio.run(run())

    assert quantity == (_GEMINI_QUANTITY, _GEMINI_USAGE)
    assert (result.binder, result.fallback_reason, usage) == ("gemini", reason, _GEMINI_USAGE)
    assert result.output.chosen_shape_index == 0
    assert step.calls == 1
    assert record.reasons == [reason, reason]
    lines = _binder_parts(caplog)
    assert [line.split(" ")[1:5] for line in lines] == [
        ["call=quantity", "binder=jev", "served_by=gemini", f"fallback={reason}"],
        ["call=bind", "binder=jev", "served_by=gemini", f"fallback={reason}"],
    ]
    defects = [r for r in caplog.records if r.levelno == logging.ERROR]
    if isinstance(fail, JevError | RequestTooLarge):
        assert defects == []
    else:
        assert [(r.exc_info or (None,))[0] for r in defects] == [type(fail)] * 2


def test_a_defect_after_the_house_policies_ran_is_raised_not_bound_twice(monkeypatch):
    """A Jev bind that raises once the house policies chose on its election re-raises, so
    no Gemini election runs them a second time."""

    def assemble_then_fail(menu, answers, *, stated_form, **_kwargs):
        menu.request.apply_policies(PolicyPool(measuring=(0,)))
        raise RuntimeError("a defect after the policies ran")

    monkeypatch.setattr(jev_module, "bind_result", assemble_then_fail)
    gemini = _GeminiDouble(_ELECTS_FIRST)
    binder = FallbackBinder(primary=JevBinder(ask=FakeAsk(_GOLDEN.stored)), fallback=gemini)
    step = _CountingStep()

    with pytest.raises(RuntimeError, match="after the policies ran"):
        asyncio.run(binder.bind(_golden_request(step), llm=None))
    assert (step.calls, gemini.requests) == (1, [])


def test_a_cancelled_jev_call_is_not_served_by_the_fallback():
    gemini = _GeminiDouble(_ELECTS_FIRST)
    binder = FallbackBinder(
        primary=JevBinder(ask=FakeAsk(fail=asyncio.CancelledError())), fallback=gemini
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(binder.read_quantity(_QUANTITY, llm=None))
    assert gemini.quantity_calls == 0


def test_a_jev_answer_is_served_without_the_fallback(caplog):
    gemini = _GeminiDouble(_ELECTS_FIRST)
    binder = FallbackBinder(primary=JevBinder(ask=FakeAsk(_GOLDEN.stored)), fallback=gemini)
    step = _CountingStep()
    caplog.set_level(logging.INFO, logger="search.engine.binders.part_log")

    with recording_fallbacks() as record:
        result, _usage = asyncio.run(binder.bind(_golden_request(step), llm=None))

    assert (result.binder, result.fallback_reason) == ("jev", None)
    assert gemini.requests == []
    assert record.reasons == []
    assert step.calls == (0 if result.output.ask else 1)
    (line,) = _binder_parts(caplog)
    assert "served_by=jev fallback=- " in line
    assert "jev_input_tokens=120 jev_output_tokens=4" in line


# -- the shadow --------------------------------------------------------------------------


def _shadow_lines(caplog: pytest.LogCaptureFixture) -> list[dict]:
    prefix = "binder_shadow "
    return [
        json.loads(r.getMessage().removeprefix(prefix))
        for r in caplog.records
        if r.getMessage().startswith(prefix)
    ]


def _draws(*values: float):
    """A `draw` returning `values` in turn."""
    it = iter(values)
    return lambda: next(it)


def test_the_shadow_answers_with_gemini_without_waiting_for_jev(caplog):
    """A sampled call returns Gemini's answer while Jev is still held, so Jev is not on the
    answer's path; the Jev call runs its policies on its own step, and once Jev answers
    the comparison is logged with the part's join keys and Jev's request, and the
    ``binder_part`` line with Jev's tokens."""
    gemini = _GeminiDouble(_ELECTS_FIRST)
    caplog.set_level(logging.INFO, logger="search.engine.binders")
    request = dataclasses.replace(_golden_request(_CountingStep()), part=2, pool="ours")

    async def run() -> tuple[tuple[BindResult, dict | None], int, bool, int]:
        gate = asyncio.Event()
        ask = FakeAsk(_GOLDEN.stored, gate=gate)
        binder = ShadowBinder(
            served=gemini, shadow=JevBinder(ask=ask), sample=0.5, draw=_draws(0.0, 0.5)
        )
        step = _CountingStep()
        answer = await binder.bind(dataclasses.replace(request, apply_policies=step), llm=None)
        held = len(ask.requests) == 1 and not gate.is_set()
        calls_at_answer = step.calls
        lines_at_answer = len(_shadow_lines(caplog)) + len(_binder_parts(caplog))
        gate.set()
        for _ in range(5):
            await asyncio.sleep(0)
        return answer, calls_at_answer, held, lines_at_answer

    (result, usage), calls, held, lines_at_answer = asyncio.run(run())

    assert held
    assert lines_at_answer == 0
    assert (result.binder, usage, calls) == ("gemini", _GEMINI_USAGE, 1)
    assert result.output.chosen_shape_index == 0
    (line,) = _shadow_lines(caplog)
    menu = [shape.shape_id for shape in request.shapes]
    assert {k: line[k] for k in ("call", "part", "pass", "control", "shadow_binder")} == {
        "call": "bind",
        "part": 2,
        "pass": "ours",
        "control": False,
        "shadow_binder": "jev",
    }
    assert (line["phrase"], line["menu"], line["failure"]) == (request.variable, menu, None)
    assert line["served"]["shape"] == menu[0]
    assert isinstance(line["agree"], bool)
    assert (line["input_tokens"], line["output_tokens"], line["chunks"]) == (120, 4, 1)
    assert line["questions"] == len(bind_request(jev_menu(request))[1])
    (part,) = _binder_parts(caplog)
    assert "binder=shadow served_by=gemini fallback=- " in part
    assert "jev_input_tokens=120 jev_output_tokens=4" in part


def test_an_unsampled_call_sends_nothing_to_jev(caplog):
    gemini = _GeminiDouble(_ELECTS_FIRST)
    ask = FakeAsk(_GOLDEN.stored)
    binder = ShadowBinder(served=gemini, shadow=JevBinder(ask=ask), sample=0.2, draw=lambda: 0.2)
    caplog.set_level(logging.INFO, logger="search.engine.binders")

    answer = asyncio.run(binder.read_quantity(_QUANTITY, llm=None))

    assert answer == (_GEMINI_QUANTITY, _GEMINI_USAGE)
    assert ask.requests == []
    assert _shadow_lines(caplog) == []
    (part,) = _binder_parts(caplog)
    assert "binder=shadow served_by=gemini" in part
    assert "jev_input_tokens=- " in part


def test_a_failed_shadow_call_leaves_the_answer_as_gemini_gives_it(caplog):
    gemini = _GeminiDouble(_ELECTS_FIRST, delay=0.01)
    ask = FakeAsk(fail=JevError(503, "server"))
    binder = ShadowBinder(
        served=gemini, shadow=JevBinder(ask=ask), sample=1.0, draw=_draws(0.0, 0.5)
    )
    caplog.set_level(logging.INFO, logger="search.engine.binders")
    request = dataclasses.replace(_QUANTITY, part=0, pool="un_system")

    answer = asyncio.run(binder.read_quantity(request, llm=None))

    assert answer == (_GEMINI_QUANTITY, _GEMINI_USAGE)
    (line,) = _shadow_lines(caplog)
    assert (line["call"], line["part"], line["pass"]) == ("quantity", 0, "un_system")
    assert (line["agree"], line["failure"], line["shadow"]) == (None, "5xx", None)
    assert line["denominators"] == [ref.ref for ref in _DENOMINATORS]
    assert (line["phrase"], line["bind_phrase"]) == (_QUANTITY.variable, _QUANTITY.bind_variable)
    (part,) = _binder_parts(caplog)
    assert "fallback=- jev_error=- " in part


class _FailsSecondQuantityRead:
    """An LLM answering the first quantity read and failing every later one."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(self, *, prompt, system, schema):
        self.calls += 1
        if self.calls > 1:
            raise LLMInfraError("control read failed")
        return QuantityOutput(stated_form="rate"), _GEMINI_USAGE


def test_a_failed_control_read_leaves_the_served_part_whole():
    """The control's quantity read fails and notes a degraded read inside its own task;
    the served part's record, which the task's context copy still points at, stays empty."""
    llm = _FailsSecondQuantityRead()
    binder = ShadowBinder(
        served=GeminiBinder(),
        shadow=JevBinder(ask=FakeAsk()),
        sample=1.0,
        control=1.0,
        draw=_draws(0.0, 0.0),
    )

    async def run():
        with recording_degraded_reads() as served:
            answer = await binder.read_quantity(_QUANTITY, llm=llm)
            for _ in range(5):
                await asyncio.sleep(0)
        return answer, served.reads

    (output, _usage), reads = asyncio.run(run())

    assert llm.calls == 2
    assert output.stated_form == "rate"
    assert reads == []


def test_a_failed_jev_shadow_call_leaves_the_served_part_whole():
    async def fail_with_a_lost_read(state, questions, *, settings):
        note_degraded_read("jev read")
        raise JevError(503, "server")

    binder = ShadowBinder(
        served=_GeminiDouble(_ELECTS_FIRST),
        shadow=JevBinder(ask=fail_with_a_lost_read),
        sample=1.0,
        draw=_draws(0.0, 0.5),
    )

    async def run():
        with recording_degraded_reads() as served:
            answer = await binder.read_quantity(_QUANTITY, llm=None)
            for _ in range(5):
                await asyncio.sleep(0)
        return answer, served.reads

    answer, reads = asyncio.run(run())

    assert answer == (_GEMINI_QUANTITY, _GEMINI_USAGE)
    assert reads == []


class _TwiceGemini(_GeminiDouble):
    """Gemini electing `outputs` in turn, spending its usage into the request's record
    (`recording_llm_usage`) on every call, and keeping the LLM each call was given."""

    def __init__(self, *outputs: _UniversalBindOutput) -> None:
        super().__init__(outputs[0])
        self._outputs = list(outputs)
        self.llms: list[object] = []

    async def bind(self, request, *, llm):
        self.llms.append(llm)
        output = self._outputs.pop(0)
        llm_client._note_spent(_GEMINI_USAGE)
        await asyncio.sleep(0)
        # `FixedBinder.bind` reads `_output` without awaiting, so the two calls never mix.
        self._output = output
        return await FixedBinder.bind(self, request, llm=llm)


def test_the_control_asks_gemini_again_and_never_jev_or_the_answer(caplog):
    """On the control share the second call is Gemini's, on an LLM without the memo, and
    it changes neither the answer, its usage, nor the served call's policy step."""
    gemini = _TwiceGemini(_ELECTS_FIRST, _UniversalBindOutput(chosen_shape_index=1))
    ask = FakeAsk(_GOLDEN.stored)
    binder = ShadowBinder(
        served=gemini,
        shadow=JevBinder(ask=ask),
        sample=0.5,
        control=0.05,
        draw=_draws(0.0, 0.04),
    )
    caplog.set_level(logging.INFO, logger="search.engine.binders")
    step = _CountingStep()
    llm = llm_client.LLM(memo=object())

    async def run():
        with llm_client.recording_llm_usage() as spent:
            answer = await binder.bind(_golden_request(step), llm=llm)
            for _ in range(5):
                await asyncio.sleep(0)
        return answer, spent.calls

    (result, usage), spent = asyncio.run(run())

    assert ask.requests == []
    assert (result.binder, result.output.chosen_shape_index, usage) == ("gemini", 0, _GEMINI_USAGE)
    assert step.calls == 1
    assert spent == [_GEMINI_USAGE]
    assert [type(x) for x in gemini.llms] == [llm_client.LLM, llm_client.LLM]
    served_llm, control_llm = gemini.llms
    assert served_llm is llm and control_llm._memo is None
    (line,) = _shadow_lines(caplog)
    assert (line["control"], line["shadow_binder"], line["agree"]) == (True, "gemini", False)
    assert line["served"]["shape"] == line["menu"][0]
    assert line["shadow"]["shape"] == line["menu"][1]
    assert (line["input_tokens"], line["questions"], line["chunks"]) == (7, None, None)
    (part,) = _binder_parts(caplog)
    assert "jev_ms=- " in part


# -- the Gemini binder's quantity read ---------------------------------------------------


class _QuantityRecordingLLM:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, type]] = []

    async def generate_structured(self, *, prompt, system, schema):
        self.requests.append((prompt, system, schema))
        return QuantityOutput(), None


def test_the_gemini_quantity_read_sends_what_it_sent_before_the_request_type():
    """The Gemini binder reads the request's phrase and denominators alone: its prompt,
    system text and schema are the ones `read_stated_quantity` sends for them."""
    llm = _QuantityRecordingLLM()

    asyncio.run(GeminiBinder().read_quantity(_QUANTITY, llm=llm))

    assert llm.requests == [
        (
            build_quantity_prompt("debt to GDP", _DENOMINATORS),
            _QUANTITY_SYSTEM_PROMPT,
            QuantityOutput,
        )
    ]


# -- usage -------------------------------------------------------------------------------


def test_gemini_usage_sums_to_the_same_three_keys():
    total = _sum_usage(_GEMINI_USAGE, {"input_tokens": 5, "output_tokens": 1, "cached_tokens": 2})

    assert json.dumps(total) == ('{"input_tokens": 12, "output_tokens": 4, "cached_tokens": 2}')


def test_jev_tokens_survive_a_sum_with_a_gemini_read():
    """A part whose quantity read Gemini served and whose bind Jev served reports both,
    Jev's under its own keys."""
    jev = JevCall(ms=200, questions=40, chunks=1, input_tokens=120, output_tokens=4).usage

    assert _sum_usage(_GEMINI_USAGE, jev) == {
        "input_tokens": 7,
        "output_tokens": 3,
        "cached_tokens": 0,
        "jev_input_tokens": 120,
        "jev_output_tokens": 4,
    }
    assert _sum_usage(_sum_usage(jev, _GEMINI_USAGE), jev)["jev_input_tokens"] == 240
