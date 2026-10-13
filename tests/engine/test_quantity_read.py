"""The quantity read's prompt, the bind call it stays out of and runs beside, and the
bases it names."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from search.engine.bind import _UNIVERSAL_BIND_SYSTEM_PROMPT, _UniversalBindOutput
from search.engine.errors import LLMInfraError
from search.engine.quantity_read import QuantityOutput, build_quantity_prompt
from search.models import GraphRef
from tests.engine import _harness
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeLLM
from tests.fixtures.keys import schema_fingerprint


def test_the_prompt_numbers_the_menus_denominators_then_none_of_these():
    """`denominator_index` indexes this list; "none of these" is always last."""
    prompt = build_quantity_prompt(
        "widget to GDP",
        [
            GraphRef(ref="Amount_GDP", label="Gross domestic product"),
            GraphRef(ref="Amount_GNI", label="Gross national income"),
        ],
    )
    assert prompt == (
        "Variable phrase: widget to GDP\n"
        "\n"
        "DENOMINATORS (the bases the candidate series divide by, other than people and area):\n"
        "  0: Gross domestic product (Amount_GDP)\n"
        "  1: Gross national income (Amount_GNI)\n"
        "  2: none of these\n"
    )


def test_the_bind_schema_and_system_prompt_carry_no_quantity_reading():
    """The bind schema's fingerprint and system prompt hash are those dev's recorded bind
    fixtures carry, so the quantity read leaves every bind fixture key unchanged.
    `test_bind.test_no_dropped_phrases_keeps_a_recall_sourced_prompt_byte_identical_to_head`
    pins the user prompt."""
    assert schema_fingerprint(_UniversalBindOutput) == "ede71de59275"
    assert (
        hashlib.sha1(_UNIVERSAL_BIND_SYSTEM_PROMPT.encode()).hexdigest()
        == "d40810528851e2109284ca3e184b748b1529eb66"
    )


class _BesideBindLLM(FakeLLM):
    """The recorded bank, with a bind call that starts only once the quantity read has
    started, and a quantity read that answers no form once the bind call lets it. With
    `bind_fails` the bind call fails, as a transport error does, and the quantity read
    waits until it is cancelled. `events` records each call's start and end."""

    def __init__(self, *, bind_fails: bool = False) -> None:
        super().__init__()
        self.bind_fails = bind_fails
        self.events: list[str] = []
        self.quantity_started: asyncio.Event | None = None

    def _started(self) -> asyncio.Event:
        if self.quantity_started is None:
            self.quantity_started = asyncio.Event()
        return self.quantity_started

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            self.events.append("quantity-start")
            self._started().set()
            try:
                await asyncio.sleep(3600 if self.bind_fails else 0)
            except asyncio.CancelledError:
                self.events.append("quantity-cancelled")
                raise
            self.events.append("quantity-end")
            return NO_QUANTITY
        if schema is _UniversalBindOutput:
            self.events.append("bind-start")
            # A bind awaited before the quantity read starts times out here.
            await asyncio.wait_for(self._started().wait(), timeout=2)
            if self.bind_fails:
                self.events.append("bind-failed")
                raise LLMInfraError("the bind call failed")
            output = await super().generate_structured(prompt=prompt, system=system, schema=schema)
            self.events.append("bind-end")
            return output
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def test_the_quantity_read_runs_beside_the_bind_call():
    """Both calls are in flight before either returns."""
    llm = _BesideBindLLM()
    response = offline_resolve(make_request("fertility rate in Kenya"), llm=llm)
    assert response.status == "definite"
    assert set(llm.events[:2]) == {"bind-start", "quantity-start"}
    assert sorted(llm.events) == ["bind-end", "bind-start", "quantity-end", "quantity-start"]


def test_a_failed_bind_call_cancels_and_awaits_the_quantity_read(monkeypatch):
    """The bind call fails while the quantity read runs: the part fails with it, and the
    quantity read is cancelled before the request returns, so no task outlives it to be
    cancelled when the event loop closes."""
    llm = _BesideBindLLM(bind_fails=True)
    resolve = _harness.resolve_async

    async def _resolve_then_return(*args, **kwargs):
        try:
            return await resolve(*args, **kwargs)
        finally:
            llm.events.append("request-returned")

    monkeypatch.setattr(_harness, "resolve_async", _resolve_then_return)
    with pytest.raises(LLMInfraError):
        offline_resolve(make_request("fertility rate in Kenya"), llm=llm)
    assert llm.events == [
        "bind-start",
        "quantity-start",
        "bind-failed",
        "quantity-cancelled",
        "request-returned",
    ]


def _answers(response) -> list[dict]:
    """`response`'s parts, less their timings."""
    return [part.model_dump(mode="json", exclude={"timing_ms"}) for part in response.parts]


class _StatedQuantityLLM(FakeLLM):
    """The recorded bank, with every quantity read answering `quantity`."""

    def __init__(self, quantity: QuantityOutput) -> None:
        super().__init__()
        self.quantity = quantity

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return self.quantity, None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


@pytest.mark.parametrize(
    ("query", "quantity"),
    [
        (
            "female literacy rate in Bangladesh",
            QuantityOutput(denominator_index=6, stated_form="rate"),
        ),
        (
            "maternal mortality ratio",
            QuantityOutput(base_words="live births", denominator_index=4, stated_form="rate"),
        ),
        (
            "poverty headcount ratio below national poverty line in Ethiopia",
            QuantityOutput(base_words="ratio", denominator_index=0, stated_form="rate"),
        ),
    ],
    ids=["female-literacy-none-of-these", "maternal-mortality-unquoted", "poverty-form-word"],
)
def test_a_base_the_read_names_on_no_words_of_the_phrase_is_no_base(query, quantity):
    """Each quantity answer recorded on these phrases names a base the phrase does not:
    "none of these" on no quoted words, a listed base on words the phrase lacks, and
    "none of these" on the form word "ratio". Each reads as the same form with no base,
    so no part refuses `denominator_not_available`. A quote of form words alone on a fake
    menu is
    `test_per_capita_guard_unresolved_denominator.test_a_quote_of_form_words_alone_names_no_base`."""
    stated = offline_resolve(make_request(query), llm=_StatedQuantityLLM(quantity))
    unnamed = offline_resolve(
        make_request(query),
        llm=_StatedQuantityLLM(QuantityOutput(stated_form=quantity.stated_form)),
    )
    assert _answers(stated) == _answers(unnamed)
    assert [part.no_data.reason for part in stated.parts if getattr(part, "no_data", None)] in (
        [],
        ["qualifier_not_bound"],
    )
