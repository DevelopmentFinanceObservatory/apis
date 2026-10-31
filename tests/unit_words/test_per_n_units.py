"""Every per-N unit the graphs declare is a rate in the unit registry or a named exclusion.

`scripts/record_per_n_units.py` writes the recorded list, so a new unit fails here until the
registry gains its row or this file says why it has none.
"""

from __future__ import annotations

import json
from pathlib import Path

from dfo_spec_kit.units import Rate, parse_unit

_RECORDED: dict[str, str | None] = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "per_n_units.json").read_text(encoding="utf-8")
)
_NO_SERIES = "no series at any entity read"
_NO_COUNT_BASE: dict[str, str] = {
    "BirthsPer1000Persons": _NO_SERIES,
    "DeathPer1000Births": _NO_SERIES,
    "DeathsPer1000Persons": _NO_SERIES,
    "MilligramsPer16Hours": "a dose over hours",
    "MilligramsPer24Hours": "a dose over hours",
    "MilligramsPer9Hours": "a dose over hours",
    "Per1000Males": _NO_SERIES,
    "PerMillionPerson": _NO_SERIES,
    "PhonesPer1000Persons": _NO_SERIES,
    "SDG_PER_100_POP": _NO_SERIES,
    "undata/UNIT_MEASURE-RATIO_COUNT_PER_100_AREA_M2": "an area, not a count",
    "undata/UNIT_MEASURE-RATIO_COUNT_PER_1_CR_USD_GNI_PER_COUNT_POP": "a base of one dollar",
}


def test_every_recorded_unit_is_a_rate_or_a_named_exclusion():
    unread = [
        ref for ref in _RECORDED if isinstance(parse_unit(ref), Rate) == (ref in _NO_COUNT_BASE)
    ]
    assert unread == []


def test_every_exclusion_is_a_recorded_unit():
    assert set(_NO_COUNT_BASE) <= set(_RECORDED)
