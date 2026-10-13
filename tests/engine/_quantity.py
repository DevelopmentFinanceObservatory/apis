"""The quantity read's answer for LLM doubles that script only the bind call."""

from __future__ import annotations

from search.engine.quantity_read import QuantityOutput

# A quantity read that finds no form and no denominator, with no usage: what a double
# returns for `QuantityOutput` when its test reads no quantity.
NO_QUANTITY: tuple[QuantityOutput, None] = (QuantityOutput(), None)
