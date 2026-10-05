"""Checks humanization of raw, camelCase, and mixed provenance labels."""

from __future__ import annotations

import pytest

from search.models import BindingValue, Slot, SlotKey, SlotValue
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import _ref, _rendered, _spec, _unbound_source_slot, assert_round_trip

# Humanize raw provenance names.


@pytest.mark.parametrize(
    ("dcid", "label", "expected_token", "expected_sentence"),
    [
        pytest.param(
            "dc/base/Kenya_Census",
            "Kenya_Census",
            "Kenya Census",
            "Development finance flow. Kenya Census.",
            id="underscore",
        ),
        pytest.param(
            "dc/base/WDI",
            "WorldDevelopmentIndicators",
            "World Development Indicators",
            "Development finance flow. World Development Indicators.",
            id="camelcase",
        ),
        pytest.param(
            "dc/base/WorldBank_WDI",
            "WorldBank_WDI",
            "World Bank WDI",
            "Development finance flow. World Bank WDI.",
            id="mixed-underscore-camelcase",
        ),
    ],
)
def test_resolved_source_label_is_humanized(
    dcid: str, label: str, expected_token: str, expected_sentence: str
):
    spec = _spec(
        slots=[_unbound_source_slot()],
        resolved_sources=[_ref(dcid, label)],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    source_tok = next(t for t in plan if t.axis == "source")
    assert source_tok.text == expected_token
    assert sentence_from_plan(plan) == expected_sentence


@pytest.mark.parametrize(
    ("dcid", "label", "expected"),
    [
        pytest.param(
            "dc/base/Kenya_Census",
            "Kenya_Census",
            "Development finance flow. Kenya Census.",
            id="underscore",
        ),
        pytest.param(
            "dc/base/WDI",
            "WorldDevelopmentIndicators",
            "Development finance flow. World Development Indicators.",
            id="camelcase",
        ),
        pytest.param(
            "dc/base/WorldBank_WDI",
            "WorldBank_WDI",
            "Development finance flow. World Bank WDI.",
            id="mixed-underscore-camelcase",
        ),
    ],
)
def test_bound_source_slot_label_is_humanized_same_as_resolved_fallback(
    dcid: str, label: str, expected: str
):
    source_slot = Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingValue(value=SlotValue(ref=_ref(dcid, label), value_kind="source")),
    )
    assert _rendered(_spec(slots=[source_slot])) == expected


@pytest.mark.parametrize(
    ("dcid", "label", "expected"),
    [
        pytest.param("worldBank/WDI", "World Bank WDI", "World Bank WDI", id="authored-label"),
        pytest.param("dc/base/WDI", "WDI", "WDI", id="all-caps-acronym"),
        pytest.param(
            "dc/base/OECDQGDP",
            "OECDQuarterlyGDP",
            "OECD Quarterly GDP",
            id="acronym-to-capitalized-word",
        ),
        pytest.param(
            "dc/base/CensusACS5",
            "CensusACS5YearSurvey",
            "Census ACS5 Year Survey",
            id="acronym-with-digit",
        ),
        pytest.param(
            "dc/base/BEAGDPv2",
            "BEAGDPv2",
            "BEAGDPv2",
            id="acronym-version-suffix",
        ),
        pytest.param(
            "dc/base/CDCWonder",
            "CDCWonderNNDSSInfectiousAnnual",
            "CDC Wonder NNDSS Infectious Annual",
            id="multiple-acronym-runs",
        ),
        pytest.param("dc/base/WorldBank", "World Bank", "World Bank", id="already-spaced"),
        pytest.param("dc/base/WDI_GDP", "WDI_GDP", "WDI GDP", id="underscore-acronyms"),
    ],
)
def test_resolved_source_identifier_humanization(dcid: str, label: str, expected: str):
    """Humanize supported acronym, digit, and word boundaries."""
    spec = _spec(
        slots=[_unbound_source_slot()],
        resolved_sources=[_ref(dcid, label)],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    source_tok = next(t for t in plan if t.axis == "source")
    assert source_tok.text == expected
