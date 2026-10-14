"""The quantity read's prompt, the bind call it stays out of and runs before, and the
bases it names."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import (
    _UNIVERSAL_BIND_SYSTEM_PROMPT,
    _build_universal_bind_prompt,
    _UniversalBindOutput,
)
from search.engine.quantity_read import QuantityOutput, build_quantity_prompt
from search.models import GraphRef
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_bind import _shape
from tests.fixtures import FakeLLM


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


def test_the_bind_schema_and_prompts_carry_no_quantity_reading():
    """The quantity read is the reading's one carrier: bind's schema has none of its fields,
    and neither bind prompt lists the denominators or asks for the reading."""
    assert not set(QuantityOutput.model_fields) & set(_UniversalBindOutput.model_fields)
    prompt = _build_universal_bind_prompt("widget to GDP", [_shape("widget")])
    for text in (_UNIVERSAL_BIND_SYSTEM_PROMPT, prompt):
        assert "DENOMINATORS" not in text
        assert "PART 3" not in text


class _OrderLLM(FakeLLM):
    """The recorded bank, with a quantity read that states no form, recording when each
    call starts and ends in `events`."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            self.events.append("quantity-start")
            await asyncio.sleep(0)
            self.events.append("quantity-end")
            return NO_QUANTITY
        if schema is _UniversalBindOutput:
            self.events.append("bind-start")
            output = await super().generate_structured(prompt=prompt, system=system, schema=schema)
            self.events.append("bind-end")
            return output
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def test_the_quantity_read_returns_before_the_bind_call_starts():
    """Admissibility refuses on the form the read states, so the read ends before the bind
    menu exists and the part waits for both calls in turn."""
    llm = _OrderLLM()
    response = offline_resolve(make_request("fertility rate in Kenya"), llm=llm)
    assert response.status == "definite"
    assert llm.events == ["quantity-start", "quantity-end", "bind-start", "bind-end"]


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
