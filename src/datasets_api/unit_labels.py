"""The `unit_label` a long download writes beside each split unit (`materialise.projection`).

A currency reads as its price basis and plural noun, the byline's unit token in sentence case:
"Current US dollars", "Constant 2023 US dollars". The nouns are Search's
(`search.heading.currency_nouns`), so a download names a currency as the answer above it does.
Any other unit reads as its graph name, else as `UNNAMED_UNIT_LABELS` for the units the graph
leaves unnamed, else stays empty.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from materialise.facets import native_currency_iso
from materialise.projection import NativeCurrency, UnitParts
from search.currency_codes import ISO_4217_CODES
from search.heading.currency_nouns import currency_noun
from search.heading.price_basis import PriceBasis, price_basis_phrase
from search.heading.unit_class import classify_unit

UNNAMED_UNIT_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {"Percent": "Percent", "SDG_PERCENT": "Percent"}
)
"""Labels of unit nodes that carry no graph `name`."""


def unit_currency(unit: str) -> NativeCurrency | None:
    """The currency of a native unit node, classified as Search classifies it
    (`classify_unit`). The node names its ISO 4217 code with no graph read when it is
    `Currency<ISO>`, a `CurrencyUnitOfMeasure` node (`native_currency_iso`), or a base Data
    Commons `Currency` node, which the graph names by its code (`USD`). Search's declared
    currency units add the US-dollar units no graph node types, with their base year. Any
    other unit returns `None`."""
    iso = native_currency_iso(unit) or (unit if unit in ISO_4217_CODES else None)
    unit_class = classify_unit(
        unit=unit, denominator=None, currency_units={unit: iso} if iso is not None else {}
    )
    if unit_class.currency is None:
        return None
    return NativeCurrency(currency=unit_class.currency, base_year=unit_class.base_year)


def unit_label(parts: UnitParts, name: str | None) -> str | None:
    """The label of one split unit. `name` is the unit node's graph name, `None` when the
    node has none or the read did not answer."""
    if parts.currency is not None:
        noun = currency_noun(parts.currency, label=name)
        basis = (
            PriceBasis(prices=parts.price_basis, base_year=parts.base_year)
            if parts.price_basis is not None
            else None
        )
        return _sentence_case(price_basis_phrase(basis, currency_noun=noun))
    return name or UNNAMED_UNIT_LABELS.get(parts.unit)


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]
