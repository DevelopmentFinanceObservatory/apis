"""Which facet units say a series counts things.

A StatVar whose measured property is ``value`` (every SDG and UN System series) says it
counts only through the unit on its facets. The graphs name count units two ways:

- The UN System Data Commons types its units ``undata/UNIT_MEASURE`` and names each by its
  SDMX ``UNIT_MEASURE`` code. A code that is ``COUNT`` or starts ``COUNT_`` is a count
  (``COUNT_CASES``, ``COUNT_DEATHS``, ``COUNT_PERSONS``); a ratio of counts starts
  ``RATIO_COUNT_`` and an average ``AVG_COUNT_``. Of the 128 units the deployment serves,
  14 match.
- Our graph's count units are the SDG numbers and ``PeopleInThousands``, listed in
  `_GRAPH_COUNT_UNITS`. The other ``SDG_UnitOfMeasure`` and ``UnitOfMeasure`` members are
  shares, rates, money, physical quantities or indexes.

Of these, the units in `_PERSON_COUNT_UNITS` say what they count is persons, as a UN System
series, whose population type is its own, says only through its unit.
"""

from __future__ import annotations

from typing import Final

_UN_UNIT_PREFIX: Final = "undata/UNIT_MEASURE-"
_UN_COUNT_CODE: Final = "COUNT"

_GRAPH_COUNT_UNITS: Final = frozenset(
    {"SDG_NUMBER", "SDG_NUM_TH", "SDG_NUM_M", "PeopleInThousands"}
)
_PERSON_COUNT_UNITS: Final = frozenset(
    {
        f"{_UN_UNIT_PREFIX}COUNT_PERSONS",
        f"{_UN_UNIT_PREFIX}COUNT_PERSONS_F",
        "PeopleInThousands",
    }
)


def is_count_unit(unit: str | None) -> bool:
    """Whether a facet's `unit` dcid names a count. A facet with no unit names nothing."""
    if unit is None:
        return False
    if unit.startswith(_UN_UNIT_PREFIX):
        code = unit.removeprefix(_UN_UNIT_PREFIX)
        return code == _UN_COUNT_CODE or code.startswith(f"{_UN_COUNT_CODE}_")
    return unit in _GRAPH_COUNT_UNITS


def is_person_count_unit(unit: str | None) -> bool:
    """Whether a facet's `unit` dcid names a count of persons (`_PERSON_COUNT_UNITS`)."""
    return unit in _PERSON_COUNT_UNITS
