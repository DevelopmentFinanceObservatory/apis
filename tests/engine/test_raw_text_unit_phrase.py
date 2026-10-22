"""A currency or price basis typed in raw text binds as the parsed door's `unit_text`.

Extraction drops the phrase ("in constant 2024 euros") as naming no code. The door hands the
phrases that bind as one unit request to every variable's leg, the way `ParsedPart.unit_text`
reaches it, and leaves the rest dropped, each with its departure.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from entity_resolution import Admission
from search.engine.concurrency import scoped_concurrency
from search.engine.entry import raw_text
from search.engine.extract import (
    _EXTRACTION_SYSTEM_PROMPT,
    Extraction,
    ExtractionResult,
    extract,
)
from search.engine.unit.bind import BoundUnitRequest, UnitDefaultPolicy, bind_unit_text
from search.models import RawTextInput
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_raw_text_handle_close_order import (
    _fake_combined_regions,
    _params,
    _stub_region,
)
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.fixtures.keys import fixture_key

_POLICY = UnitDefaultPolicy(currency="USD", base_year=lambda: 2023)


def _prep(*dropped: str, variables: tuple[str, ...] = ("aid",)) -> raw_text._ExtractionPrep:
    extraction = Extraction(
        kind="resolution", variables=list(variables), dropped_qualifiers=list(dropped)
    )
    return raw_text._prepare_extraction(
        extraction, ExtractionResult(extraction=extraction, qualifiers={}), query="aid"
    )


@pytest.mark.parametrize(
    ("phrase", "currency", "prices", "base_year"),
    [
        pytest.param("in constant 2024 euros", "EUR", "constant", 2024, id="constant-euros"),
        pytest.param("in euros", "EUR", "current", None, id="current-euros"),
        pytest.param("in constant euros", "EUR", "constant", 2023, id="default-base-year"),
    ],
)
def test_a_typed_currency_phrase_binds_as_the_unit_text(phrase, currency, prices, base_year):
    prep = _prep(phrase)

    assert prep.dropped_qualifier_phrases == []
    assert prep.qualifier_facts == {}
    assert prep.qualifier_warnings == []
    assert prep.unit_text == phrase
    outcome = bind_unit_text(phrase, policy=_POLICY)
    assert isinstance(outcome, BoundUnitRequest)
    assert (outcome.currency, outcome.prices, outcome.base_year) == (currency, prices, base_year)


def test_a_unit_phrase_inside_the_variable_binds_as_the_unit_text():
    query = "GDP of Kenya in euros"
    llm = FakeLLM(
        responses={
            fixture_key(Extraction, _EXTRACTION_SYSTEM_PROMPT, query): {
                "kind": "resolution",
                "variables": ["GDP in euros"],
                "entities": [{"text": "Kenya", "span": "Kenya"}],
                "dates": [],
            }
        }
    )
    result, _ = asyncio.run(extract(query, llm=llm))

    prep = raw_text._prepare_extraction(result.extraction, result, query=query)

    assert prep.unique == ["GDP"]
    assert prep.per_variable_unit_texts == ["in euros"]
    assert prep.dropped_qualifier_phrases == []


def test_a_phrase_that_cannot_bind_stays_dropped_with_its_departure():
    prep = _prep("in zorkmids", "in constant 2022 dollars and euros", "through development banks")

    assert prep.unit_text is None
    assert prep.dropped_qualifier_phrases == [
        "in zorkmids",
        "in constant 2022 dollars and euros",
        "through development banks",
    ]
    assert [(fact.code, fact.asked) for fact in prep.qualifier_facts.values()] == [
        ("measure_not_applied", phrase) for phrase in prep.dropped_qualifier_phrases
    ]


def test_a_second_phrase_that_conflicts_with_the_first_stays_dropped():
    prep = _prep("in euros", "in yen", "at constant prices")

    assert prep.unit_text == "in euros at constant prices"
    assert prep.dropped_qualifier_phrases == ["in yen"]


async def _legs_unit_texts(
    extraction: Extraction, *, unit_texts: dict[str, str] | None = None
) -> list[tuple[str, str | None, bool, list[str]]]:
    legs: list[tuple[str, str | None, bool, list[str]]] = []

    async def _extract(query, *, llm):
        result = ExtractionResult(extraction=extraction, qualifiers={}, unit_texts=unit_texts or {})
        return result, None

    async def _resolve_variable(variable, **kwargs):
        legs.append(
            (
                variable,
                kwargs["unit_text"],
                kwargs["unit_text_best_effort"],
                kwargs["dropped_qualifier_phrases"],
            )
        )
        return _stub_region(variable)

    graph = FakeGraph()
    with (
        patch("search.engine.entry.raw_text.extract", _extract),
        patch("search.engine.fanout.resolve_variable", _resolve_variable),
        patch("search.engine.entry.raw_text._stream_combined_regions", _fake_combined_regions([])),
    ):
        async with scoped_concurrency():
            async for _ in raw_text.handle(
                RawTextInput(query="aid and debt in euros"),
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=_params(),
            ):
                pass
    return sorted(legs)


@pytest.mark.parametrize("variables", [["aid"], ["aid", "debt"]])
def test_every_variable_leg_receives_the_unit_text(variables):
    extraction = Extraction(kind="resolution", variables=variables, dropped_qualifiers=["in euros"])

    legs = asyncio.run(_legs_unit_texts(extraction))

    # A typed phrase is best effort, so data that cannot convert answers without it.
    assert legs == [(variable, "in euros", True, []) for variable in sorted(variables)]


@pytest.mark.parametrize(
    ("variables", "dropped", "unit_texts", "expected"),
    [
        pytest.param(
            ["GDP", "population"],
            [],
            {"GDP": "in euros"},
            {"GDP": "in euros", "population": None},
            id="own-phrase-stays-on-its-part",
        ),
        pytest.param(
            ["GDP", "debt"],
            [],
            {"GDP": "in euros", "debt": "in dollars"},
            {"GDP": "in euros", "debt": "in dollars"},
            id="each-part-its-own-currency",
        ),
        pytest.param(
            ["GDP", "debt"],
            ["at constant prices"],
            {"GDP": "in euros"},
            {"GDP": "in euros at constant prices", "debt": "at constant prices"},
            id="own-phrase-joins-the-request-wide-one",
        ),
        pytest.param(
            ["GDP", "debt"],
            ["in yen"],
            {"GDP": "in euros"},
            {"GDP": "in euros", "debt": "in yen"},
            id="own-phrase-wins-a-conflict",
        ),
    ],
)
def test_a_peeled_unit_text_reaches_only_its_own_part(variables, dropped, unit_texts, expected):
    extraction = Extraction(kind="resolution", variables=variables, dropped_qualifiers=dropped)

    legs = asyncio.run(_legs_unit_texts(extraction, unit_texts=unit_texts))

    assert {variable: unit_text for variable, unit_text, _, _ in legs} == expected
