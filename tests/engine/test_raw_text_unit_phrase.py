"""A currency or price basis typed in raw text binds as the parsed door's `unit_text`.

Extraction drops the phrase ("in constant 2024 euros") as naming no code. The door hands the
phrases that bind as one unit request to every variable's leg, the way `ParsedPart.unit_text`
reaches it, and leaves the rest dropped, each with its departure.
"""

from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from entity_resolution import Admission
from search.engine.concurrency import scoped_concurrency
from search.engine.config import settings
from search.engine.entry import raw_text
from search.engine.extract import ExtractedExclusion, Extraction, ExtractionResult
from search.engine.unit.bind import BoundUnitRequest, UnitDefaultPolicy, bind_unit_text
from search.models import RawTextInput
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_raw_text_handle_close_order import (
    _fake_combined_regions,
    _params,
    _stub_region,
)
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_POLICY = UnitDefaultPolicy(currency="USD", base_year=lambda: 2023)


def _prep(
    *dropped: str,
    variables: tuple[str, ...] = ("aid",),
    exclusions: tuple[ExtractedExclusion, ...] = (),
) -> raw_text._ExtractionPrep:
    extraction = Extraction(
        kind="resolution",
        variables=list(variables),
        dropped_qualifiers=list(dropped),
        exclusions=list(exclusions),
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


def test_a_phrase_the_query_left_out_never_binds_as_the_unit_or_reaches_bind():
    """An exclusion that names a currency is no unit request, and bind never reads it as
    something asked for; it warns and departs as left out."""
    prep = _prep(
        "in euros",
        "through development banks",
        exclusions=(ExtractedExclusion(phrase="in euros", left_out="euros"),),
    )

    assert prep.unit_text is None
    assert prep.dropped_qualifier_phrases == ["through development banks"]
    assert [w.message for w in prep.exclusion_warnings] == [
        'The requested qualifier "in euros" was not applied.'
    ]
    assert [(f.asked, f.left_out) for f in prep.qualifier_facts.values()] == [
        ("through development banks", False),
        ("in euros", True),
    ]


def test_a_second_phrase_that_conflicts_with_the_first_stays_dropped():
    prep = _prep("in euros", "in yen", "at constant prices")

    assert prep.unit_text == "in euros at constant prices"
    assert prep.dropped_qualifier_phrases == ["in yen"]


async def _leg_arguments(
    extraction: Extraction,
    *,
    query: str = "aid and debt in euros",
    request_text: str | None = None,
    scope: object | None = None,
) -> list[tuple[str, dict]]:
    """Each leg's variable and the keyword arguments `handle` resolves it with."""
    legs: list[tuple[str, dict]] = []

    async def _extract(query, *, llm):
        result = ExtractionResult(extraction=extraction, qualifiers={}, request_text=request_text)
        return result, None

    async def _resolve_variable(variable, **kwargs):
        legs.append((variable, kwargs))
        return _stub_region(variable)

    graph = FakeGraph()
    with (
        patch("search.engine.entry.raw_text.extract", _extract),
        patch("search.engine.fanout.resolve_variable", _resolve_variable),
        patch("search.engine.entry.raw_text._stream_combined_regions", _fake_combined_regions([])),
    ):
        async with scoped_concurrency():
            async for _ in raw_text.handle(
                RawTextInput(query=query),
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=_params(),
                scope=scope,
            ):
                pass
    return sorted(legs, key=lambda leg: leg[0])


@pytest.mark.parametrize("variables", [["aid"], ["aid", "debt"]])
def test_every_variable_leg_receives_the_unit_text(variables):
    extraction = Extraction(kind="resolution", variables=variables, dropped_qualifiers=["in euros"])

    legs = asyncio.run(_leg_arguments(extraction))

    # A typed phrase is best effort, so data that cannot convert answers without it.
    assert [
        (
            variable,
            kwargs["unit_text"],
            kwargs["unit_text_best_effort"],
            kwargs["dropped_qualifier_phrases"],
        )
        for variable, kwargs in legs
    ] == [(variable, "in euros", True, []) for variable in sorted(variables)]


def test_a_leg_matches_the_request_text_and_judges_the_phrases_left_out():
    """The leg never sees the stretch an exclusion covers: its match text is the request
    text, and the phrase reaches it only as something to judge its answers on, with every
    thing it leaves out."""
    query = "aid to Kenya excluding humanitarian aid and loans in 2022"
    phrase = "excluding humanitarian aid and loans"
    exclusions = [
        ExtractedExclusion(phrase=phrase, left_out="humanitarian aid"),
        ExtractedExclusion(phrase=phrase, left_out="loans"),
    ]
    extraction = Extraction(
        kind="resolution", variables=["aid"], dropped_qualifiers=[phrase], exclusions=exclusions
    )

    scope = MagicMock(lookup_parsed=AsyncMock(return_value=None))
    [(_, kwargs)] = asyncio.run(
        _leg_arguments(extraction, query=query, request_text="aid to Kenya ; in 2022", scope=scope)
    )

    # The cache signals read what the leg reads.
    looked_up = scope.lookup_parsed.await_args.kwargs
    assert (looked_up["query_text"], looked_up["exclusions"]) == (
        "aid to Kenya ; in 2022",
        exclusions,
    )
    assert kwargs["query_text"] == "aid to Kenya ; in 2022"
    assert kwargs["exclusions"] == exclusions
    assert kwargs["dropped_qualifier_phrases"] == []


@pytest.mark.parametrize(
    ("met", "warned"),
    [
        ([({"not net"}, True)], False),
        ([(set(), True)], True),
        # Every leg's answers must leave the phrase out.
        ([({"not net"}, True), (set(), True)], True),
        ([], True),
        # A leg that refuses shows nothing that leaves the phrase out, whatever it judged.
        ([({"not net"}, True), ({"not net"}, False)], True),
    ],
)
def test_an_exclusion_every_answer_meets_makes_no_claim(met, warned):
    prep = _prep(
        "not net",
        variables=("gross ODA",),
        exclusions=(ExtractedExclusion(phrase="not net", left_out="net"),),
    )
    regions = [
        dataclasses.replace(
            _stub_region("gross ODA"),
            met_exclusions=frozenset(held),
            **({"status": "definite", "specs": (MagicMock(),)} if answered else {}),
        )
        for held, answered in met
    ]

    assert [w.message for w in prep.surviving(regions)] == (
        ['The requested qualifier "not net" was not applied.'] if warned else []
    )


def test_answers_are_judged_on_the_whole_phrase_past_its_display_length():
    """The dropped qualifiers show a long phrase cut to the display length; the warning
    quotes that form, and the legs judge their answers on the whole phrase."""
    limit = settings().max_dropped_qualifier_length
    phrase = "excluding " + "humanitarian aid and " * (limit // 20) + "loans"
    extraction = Extraction(
        kind="resolution",
        variables=["aid"],
        dropped_qualifiers=[phrase[:limit]],
        exclusions=[ExtractedExclusion(phrase=phrase, left_out="loans")],
    )
    prep = raw_text._prepare_extraction(
        extraction, ExtractionResult(extraction=extraction, qualifiers={}), query="aid"
    )

    assert len(phrase) > limit
    assert prep.exclusion_phrases == [phrase]
    assert [w.message for w in prep.exclusion_warnings] == [
        f'The requested qualifier "{phrase[:limit]}" was not applied.'
    ]
