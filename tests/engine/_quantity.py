"""The quantity read's answer for LLM doubles that script only the bind call, and a double
for tests that stop before bind."""

from __future__ import annotations

from search.engine.quantity_read import QuantityOutput

# A quantity read that finds no form and no denominator, with no usage: what a double
# returns for `QuantityOutput` when its test reads no quantity.
NO_QUANTITY: tuple[QuantityOutput, None] = (QuantityOutput(), None)


class QuantityReadOnlyLLM:
    """A double for tests that run the stages before bind: it answers the quantity read
    with no form and fails any other call."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema is QuantityOutput, f"unexpected schema {schema.__name__!r}"
        return NO_QUANTITY
