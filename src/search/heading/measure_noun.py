"""Measure nouns for IDS and WEO series, and nouns for the denominators their ratios use.

OECD CRS and ONE imputed partitions take their partition label
(`measure_label_for_partition`). Other series take the code index's measure nouns
(`MeasureIndex.nouns`), authored beside the partition labels in
`scripts/export_code_index.py`. Without one a heading would read the graph name, which
carries the unit inside it: "Debt service on external debt (current US$)", "Gross
domestic product (GDP), Per capita, purchasing power parity (PPP) international dollar,
ICP benchmarks 2017-2021". `measure_noun` gives "external debt service" and "GDP"
instead. The unit (`search.heading.unit_class`) carries "per capita", "PPP
international dollars" and the share base.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from code_index.shared import get_measure_index
from code_index.types import label_for_pairs

_POPULATION: Final = "populationType"
_PROPERTY: Final = "measuredProperty"
_STAT_TYPE: Final = "statType"

_DENOMINATOR_NOUNS: Final[Mapping[str, str]] = {
    "Amount_EconomicActivity_ExportsOfGoodsServicesAndPrimaryIncome_Nominal": (
        "exports of goods, services and primary income"
    ),
    "Amount_EconomicActivity_GrossNationalIncome_Nominal": "GNI",
    "ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_Nominal": (
        "GDP"
    ),
    (
        "ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption"
        "_CyclicallyAdjusted_RealValue"
    ): "potential GDP",
}
"""Nouns for the `measurementDenominator` StatVars IDS and WEO ratios divide by."""


@dataclass(frozen=True, slots=True)
class MeasureNoun:
    """An authored noun, the measure axes it states, and the code pairs its text names.

    `axes` uses `ResolvedSpec.measure_label_axes`' vocabulary (`populationType`,
    `measuredProperty`, `statType`, or a constraint-property dcid). A renderer adds them
    to `measure_label_axes`, so no how clause repeats what the noun says ("for servicing
    payment" after "external debt service"). `named_values` are the entry's
    `MeasurePartitionLabel.named_values`.
    """

    noun: str
    axes: tuple[str, ...]
    named_values: tuple[tuple[str, str], ...] = ()


def measure_noun(
    *,
    population_type: str | None,
    measured_property: str | None,
    stat_type: str | None,
    constraints: Mapping[str, str],
) -> MeasureNoun | None:
    """The code index's noun for a measure (`MeasureIndex.nouns`), or `None` when no entry
    covers it.

    Args:
        population_type: the shape's `populationType` dcid.
        measured_property: the shape's `measuredProperty` dcid.
        stat_type: the shape's `statType` dcid.
        constraints: constraint values every served member shares, property dcid to
            value dcid. A property the members disagree on is left out.

    Returns:
        The most specific entry whose every pair the measure carries. GDP at purchasing
        power parity matches the entry that also states `currencyConversionBasis`, so
        that axis is consumed.
    """
    pairs = [
        (_POPULATION, population_type),
        (_PROPERTY, measured_property),
        (_STAT_TYPE, stat_type),
        *constraints.items(),
    ]
    match = label_for_pairs(pairs, get_measure_index().nouns)
    if match is None:
        return None
    uncovered = frozenset(match.uncovered)
    axes = tuple(
        axis for axis, value in pairs if value is not None and (axis, value) not in uncovered
    )
    return MeasureNoun(noun=match.label, axes=axes, named_values=match.named_values)


def denominator_noun(denominator: str | None) -> str | None:
    """The noun for a ratio's `measurementDenominator` StatVar dcid, or `None`.

    Per-head denominators (`Count_Person`, `PerCapita`) have none, since the unit class states
    them as "per capita".
    """
    if denominator is None:
        return None
    return _DENOMINATOR_NOUNS.get(denominator)
