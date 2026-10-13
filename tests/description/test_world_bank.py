"""`world_bank_definition` over every WDI and IDS StatVar description on the dev graph.

`fixtures/world_bank_descriptions.json.gz` holds the `description` values of the 2,898
World Development Indicators StatVars (`worldBank/`) and the 1,073 International Debt
Statistics StatVars, as `scripts/record_world_bank_descriptions.py` records them. A graph read
takes a StatVar's first value (`graph_api.card_definition._first`), so these cases do too.
"""

from __future__ import annotations

import gzip
import json
from functools import cache
from pathlib import Path

import pytest

from description.world_bank import world_bank_definition

_FIXTURE = Path(__file__).parent / "fixtures" / "world_bank_descriptions.json.gz"
_CODE_NOTE = "(WorldBank Series Code: "


@cache
def _recorded() -> dict[str, dict[str, list[str]]]:
    return json.loads(gzip.decompress(_FIXTURE.read_bytes()))


def _first(group: str) -> dict[str, str]:
    return {dcid: values[0] for dcid, values in _recorded()[group].items()}


def test_every_wdi_definition_leaves_out_its_one_series_code() -> None:
    """Every WDI description ends in one series code note, 497 of them with a source note
    holding parentheses of its own. Cut, 2,788 leave a definition and 110 leave nothing, since
    their description is the note alone."""
    descriptions = _first("wdi")
    defined = {dcid: world_bank_definition(dcid, text) for dcid, text in descriptions.items()}

    assert len(descriptions) == 2_898
    assert all(text.count(_CODE_NOTE) == 1 for text in descriptions.values())
    assert not [text for text in defined.values() if text is not None and "Series Code" in text]
    assert sum(text is not None for text in defined.values()) == 2_788
    assert all(
        descriptions[dcid].lstrip().startswith(_CODE_NOTE)
        for dcid, text in defined.items()
        if text is None
    )


@pytest.mark.parametrize(
    ("dcid", "definition"),
    [
        pytest.param(
            "worldBank/1_1_ACCESS_ELECTRICITY_TOT",
            "Access to electricity (% of total population): Percentage of total population "
            "with access to electricity",
            id="code-note",
        ),
        pytest.param(
            "worldBank/AG_LND_ARBL_HA",
            "Arable land (in hectares) includes land defined by the FAO as land under temporary "
            "crops (double-cropped areas are counted once), temporary meadows for mowing or for "
            "pasture, land under market or kitchen gardens, and land temporarily fallow. Land "  # codespell:ignore fallow
            "abandoned as a result of shifting cultivation is excluded.",
            id="code-and-source-note",
        ),
        pytest.param(
            "worldBank/AG_LND_EL5M_ZS",
            "Land area below 5m is the percentage of total land where the elevation is 5 "
            "meters or less.",
            id="source-note-with-parentheses",
        ),
        pytest.param("worldBank/NYGDPMKTPKDZ", None, id="code-note-alone"),
    ],
)
def test_a_wdi_definition_is_its_description_before_the_series_code(
    dcid: str, definition: str | None
) -> None:
    assert world_bank_definition(dcid, _first("wdi")[dcid]) == definition


def test_no_ids_statvar_takes_a_definition_from_its_description() -> None:
    """IDS descriptions are series titles. Their StatVars sit outside the `worldBank/`
    namespace, and none carries the series code note either."""
    descriptions = _first("ids")

    assert len(descriptions) == 1_073
    assert not [text for text in descriptions.values() if _CODE_NOTE in text]
    assert not [
        dcid for dcid, text in descriptions.items() if world_bank_definition(dcid, text) is not None
    ]
