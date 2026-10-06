"""Pins the heading's words on specs recorded on dev: group nouns, a unit the measure's
name already states, IDS creditor types as the creditor role, how-value connectives,
the flow 30 short name, and entity names in the graph label's case.

`heading_words_specs.json` holds each recorded spec trimmed to the fields the renderer
reads; every one renders the same text as its untrimmed original. A spec's `sentence_facts`
holds what the renderer reads that the wire does not carry: its members' facts
(`ResolvedSpec.sentence_facts`) and the code pairs its measure label names
(`ResolvedSpec.measure_label_named_values`).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest

from dfo_platform.roles import role_wording
from search.render.how import _role_class_text, _without_property_suffix
from search.render.plan import render_sentence_plan, sentence_from_plan
from tests.render._helpers import recorded_spec

_SPECS = Path(__file__).with_name("heading_words_specs.json")


@cache
def _recorded() -> dict[str, dict]:
    return json.loads(_SPECS.read_text())


def _segments(key: str) -> tuple[str, str]:
    plan = render_sentence_plan(recorded_spec(_recorded()[key]))
    heading = sentence_from_plan([t for t in plan if t.segment == "heading"])
    byline = sentence_from_plan([t for t in plan if t.segment == "byline"]).strip()
    return heading, byline


@pytest.mark.parametrize(
    ("key", "heading"),
    [
        pytest.param(
            "crs-006",
            "Gross ODA disbursements to Ethiopia from the 27 European Union members, 1973 to 2024.",
            id="crs-006",
        ),
        pytest.param(
            "ids-016",
            "Kenya's external debt service owed to the 19 G20 members, 1970 to 2024.",
            id="ids-016",
        ),
        pytest.param("weo-019-0", "GDP in the 7 G7 members, 1960 to 2024.", id="weo-019"),
    ],
)
def test_a_group_reads_as_its_members(key: str, heading: str):
    """A group reads "the {n} {label} members" unless its label already ends in a noun
    for the members (`test_render_heading.py` pins "the 4 EU member states")."""
    assert _segments(key)[0] == heading


_IDS = "World Bank International Debt Statistics (IDS)"
_UNIDO = "United Nations Industrial Development Organization (UNIDO)"


@pytest.mark.parametrize(
    ("key", "heading", "byline"),
    [
        pytest.param(
            "ids-031-1",
            "Ghana's net flows on external debt for short-term maturity, 1970 to 2024.",
            f"Current US dollars · {_IDS}.",
            id="ids-031-authored-noun",
        ),
        pytest.param(
            "ids-031-0",
            "Ghana's outstanding external debt for short-term maturity, 1970 to 2024.",
            f"Current US dollars · {_IDS}.",
            id="ids-031-control",
        ),
        pytest.param(
            "ids-013",
            "IMF's flows on long-term external debt with IMF credit to Zambia, 1970 to 2024.",
            f"Current US dollars · {_IDS}.",
            id="ids-013-authored-noun",
        ),
        pytest.param(
            "weo-012-0",
            "Sub-Saharan Africa's gross domestic product at current prices in US dollars,"
            " 1990 to 2023.",
            f"{_UNIDO}.",
            id="weo-012",
        ),
        pytest.param(
            "weo-012-1",
            "Sub-Saharan Africa's gross domestic product at constant prices of 2015 in US"
            " dollars, 1990 to 2024.",
            f"{_UNIDO}.",
            id="weo-012-constant",
        ),
        pytest.param(
            "weo-017-0",
            "Gross domestic product at current prices in US dollars in the 21 Euro Area"
            " members, 1990 to 2023.",
            f"{_UNIDO}.",
            id="weo-017",
        ),
        pytest.param(
            "weo-018-0",
            "East Africa's gross domestic product at current prices in US dollars, 1990 to 2023.",
            f"{_UNIDO}.",
            id="weo-018",
        ),
        pytest.param(
            "crs-021-0",
            "Official Development Assitance (ODA) outflow in United States dollars in Africa"  # codespell:ignore
            " and United Kingdom in 2024.",
            "United Nations Children's Fund (UNICEF).",
            id="crs-021",
        ),
        pytest.param(
            "crs-021-1",
            "Outbound official development assistance (ODA) from OECD-DAC countries on grant"
            " equivalent basis in Africa and United Kingdom, 2018 to 2024.",
            "US dollars · SDG Global Database.",
            id="crs-021-name-without-unit",
        ),
    ],
)
def test_a_unit_reads_once(key: str, heading: str, byline: str):
    """An authored noun replaces a graph name carrying "(current US$)", and the byline
    leaves out a US-dollar unit the graph name the heading uses already states."""
    assert _segments(key) == (heading, byline)


@pytest.mark.parametrize(
    ("key", "heading"),
    [
        pytest.param(
            "ids-008",
            "Kenya's external debt service owed to bondholders, 1970 to 2024.",
            id="ids-008",
        ),
        pytest.param(
            "ids-010",
            "Ghana's external debt service owed to bilateral creditors, 1970 to 2024.",
            id="ids-010",
        ),
    ],
)
def test_an_ids_creditor_type_reads_as_the_creditor_role(key: str, heading: str):
    assert _segments(key)[0] == heading


@pytest.mark.parametrize(
    ("key", "heading"),
    [
        pytest.param(
            "crs-010",
            "Gates Foundation's gross ODA disbursements to Nigeria as grants.",
            id="crs-010",
        ),
        pytest.param(
            "crs-011",
            "Gates Foundation's gross private philanthropy disbursements to 142 recipients"
            " as grants, 2009 to 2024.",
            id="crs-011",
        ),
        pytest.param(
            "crs-033",
            "Japan's gross ODA disbursements to Vietnam for economic infrastructure and"
            " services as debt instruments, 1994 to 2024.",
            id="crs-033",
        ),
        pytest.param("mls-029", "Imputed multilateral ODA to Gavi.", id="mls-029"),
    ],
)
def test_how_values_take_their_property_connective(key: str, heading: str):
    """A purpose reads "for", a finance type "as" with its code's short name, and the counterpart precedes both.
    An entity keeps its graph label's case."""
    assert _segments(key)[0] == heading


# Every IDSCreditorTypeEnum member's graph name, read on dev, and the phrase it reads as.
_IDS_CREDITOR_TYPES = {
    "bilateral (IDS creditor type)": "bilateral creditors",
    "multilateral (IDS creditor type)": "multilateral creditors",
    "other multilateral (IDS creditor type)": "other multilateral creditors",
    "official creditors (IDS creditor type)": "official creditors",
    "private creditors (IDS creditor type)": "private creditors",
    "other private creditors (IDS creditor type)": "other private creditors",
    "commercial banks (IDS creditor type)": "commercial banks",
    "commercial banks or other private creditors (IDS creditor type)": (
        "commercial banks or other private creditors"
    ),
    "regional development banks (IDS creditor type)": "regional development banks",
    "bondholders (IDS creditor type)": "bondholders",
}


@pytest.mark.parametrize(("label", "phrase"), _IDS_CREDITOR_TYPES.items())
def test_every_ids_creditor_type_names_its_creditors(label: str, phrase: str):
    """The creditor-type rule matches a final "s" as text, so each of the ten graph
    names is pinned."""
    wording = role_wording("debt_service")
    bare = _without_property_suffix(label, "IDS creditor type")
    assert _role_class_text(bare, "idsCreditorType", wording) == phrase
