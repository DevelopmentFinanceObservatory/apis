"""The raw-text door marks the variables its query compares `separate`, by their own words.

`source_spans.read_sources` locates each extracted variable in the raw query and marks two
located variables a comparison connective governs. The marks reach the fan-out legs and the
layer-2 key as `ParsedPart.separate` does on the parsed door.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from unittest.mock import patch

import pytest

from entity_resolution import Admission
from search.engine.concurrency import scoped_concurrency
from search.engine.entry import raw_text
from search.engine.extract import Extraction, ExtractionResult
from search.engine.response_cache import ResponseCache
from search.engine.source_spans import (
    locate_variables,
    read_sources,
    unclaimed_words,
)
from search.models import RawTextInput
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_raw_text_handle_close_order import _params, _stub_region
from tests.engine.test_response_cache_flow import _make_scope
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_COMPARE = "compare aid to health and education"


@pytest.mark.parametrize(
    ("query", "variables", "separate"),
    [
        pytest.param(_COMPARE, ["health aid", "education aid"], (True, True), id="compare-verb"),
        pytest.param(
            "health versus education aid",
            ["health aid", "education aid"],
            (True, True),
            id="versus",
        ),
        pytest.param(
            "health and education aid",
            ["health aid", "education aid"],
            (False, False),
            id="union",
        ),
        pytest.param(
            _COMPARE, ["health aid", "schooling aid"], (False, False), id="one-not-located"
        ),
        pytest.param(
            "compare aid to health, education and water",
            ["health aid", "education aid", "water aid"],
            (True, True, True),
            id="three-compared",
        ),
        pytest.param(
            "health aid and education aid compared to 2010",
            ["health aid", "education aid"],
            (False, False),
            id="compared-to-a-year",
        ),
        pytest.param(_COMPARE, ["health aid"], (False,), id="one-variable"),
        pytest.param(
            "compare health aid and education aid",
            ["health aid", "education aid"],
            (True, True),
            id="compare-across-a-shared-word",
        ),
        pytest.param(
            "compare health aid with education aid in Kenya",
            ["health aid", "education aid"],
            (True, True),
            id="compare-a-run-joined-by-with",
        ),
        pytest.param(
            "ODA to health vs. ODA to education",
            ["ODA to health", "ODA to education"],
            (True, True),
            id="versus-before-shared-words",
        ),
        pytest.param(
            "health aid and education aid",
            ["health aid", "education aid"],
            (False, False),
            id="union-across-a-shared-word",
        ),
        pytest.param(
            "compare health ODA with basic health ODA",
            ["health ODA", "basic health ODA"],
            (True, True),
            id="compare-a-variable-whose-words-another-holds",
        ),
        pytest.param(
            "compare health ODA with basic health ODA",
            ["basic health ODA", "health ODA"],
            (True, True),
            id="compare-nested-variables-in-reverse-extraction-order",
        ),
        pytest.param(
            "health ODA vs. basic health ODA",
            ["basic health ODA", "health ODA"],
            (True, True),
            id="punctuated-versus-nested-variables-in-reverse-extraction-order",
        ),
        pytest.param(
            "health ODA vs. health ODA",
            ["health ODA", "health ODA"],
            (True, True),
            id="identical-variables-at-repeated-raw-phrases",
        ),
        pytest.param(
            "How much ODA did Kenya receive? Compare health ODA with education ODA",
            ["health ODA", "education ODA"],
            (True, True),
            id="compare-after-a-shared-word",
        ),
        pytest.param(
            "For ODA to Kenya, compare health with education",
            ["health ODA", "education ODA"],
            (True, True),
            id="compare-after-a-shared-word-each-part-one-word",
        ),
        pytest.param(
            "aid to Kenya, compared between health and education",
            ["health aid", "education aid"],
            (True, True),
            id="compared-after-a-shared-word",
        ),
        pytest.param(
            "ODA to Kenya in 2020: compare health, education and water",
            ["health ODA", "education ODA", "water ODA"],
            (True, True, True),
            id="compare-three-after-a-shared-word",
        ),
        pytest.param(
            "How much ODA did Kenya receive? Compare health ODA with basic health ODA",
            ["health ODA", "basic health ODA"],
            (True, True),
            id="compare-after-a-shared-word-a-variable-whose-words-another-holds",
        ),
        pytest.param(
            "ODA to health vs. ODA to education",
            ["ODA to education", "ODA to health"],
            (True, True),
            id="versus-variables-out-of-query-order",
        ),
        pytest.param(
            "health ODA vs education ODA",
            ["education ODA", "health ODA"],
            (True, True),
            id="versus-out-of-query-order",
        ),
        pytest.param(
            "health and education aid",
            ["education aid", "health aid"],
            (False, False),
            id="union-out-of-query-order",
        ),
    ],
)
def test_a_comparison_connective_governing_two_located_variables_marks_both(
    query, variables, separate
):
    assert read_sources(query, variables).separate == separate


@pytest.mark.parametrize(
    ("query", "variables", "spans"),
    [
        pytest.param(
            _COMPARE,
            ["health aid", "education aid", "schooling aid"],
            ["aid to health", "education", None],
            id="a-shared-word-before-both",
        ),
        pytest.param(
            "health and education aid",
            ["health aid", "education aid"],
            ["health", "education aid"],
            id="a-shared-word-after-both",
        ),
        pytest.param(
            "compare health ODA with basic health ODA",
            ["health ODA", "basic health ODA"],
            ["health ODA", "basic health ODA"],
            id="one-variable-holds-the-others-words",
        ),
        pytest.param(
            "compare health ODA with basic health ODA",
            ["basic health ODA", "health ODA"],
            ["basic health ODA", "health ODA"],
            id="one-variable-holds-the-others-words-in-reverse-extraction-order",
        ),
        pytest.param(
            "health ODA vs. health ODA",
            ["health ODA", "health ODA"],
            ["health ODA", "health ODA"],
            id="identical-variables-at-repeated-raw-phrases",
        ),
        pytest.param(
            "health and education aid",
            ["education aid", "health aid"],
            ["education aid", "health"],
            id="out-of-query-order",
        ),
        pytest.param(
            "For ODA to Kenya, compare health ODA with basic health ODA",
            ["health ODA", "basic health ODA"],
            ["health ODA", "basic health ODA"],
            id="a-shared-word-before-a-comparison-connective",
        ),
        pytest.param(
            "How much ODA did Kenya receive? Compare health ODA with education ODA",
            ["health ODA", "education ODA"],
            ["health ODA", "education ODA"],
            id="a-span-stops-at-a-comparison-connective",
        ),
    ],
)
def test_variables_align_to_the_query_words(query, variables, spans):
    """Each variable spans its own words and stops at a word only another variable holds,
    a word it has read already or a comparison connective. `spans` holds the query text
    each variable spans. Spans never overlap."""
    located = locate_variables(query, variables)
    assert [None if span is None else query[span.start : span.end] for span in located] == spans
    ordered = sorted((span.start, span.end) for span in located if span is not None)
    assert all(end <= start for (_, end), (start, _) in itertools.pairwise(ordered))


@pytest.mark.parametrize(
    ("query", "claimed", "read", "unclaimed"),
    [
        pytest.param(_COMPARE, ["health aid", "education aid"], {0}, (), id="all-claimed"),
        pytest.param(
            _COMPARE, ["health aid", "education aid"], set(), ("compare",), id="connective-unread"
        ),
        pytest.param(
            "maternal health ODA to Kenya in 2020",
            ["health ODA", "Kenya", "2020"],
            set(),
            ("maternal",),
            id="modifier",
        ),
    ],
)
def test_unclaimed_words_are_the_content_words_nothing_claims(query, claimed, read, unclaimed):
    assert unclaimed_words(query, claimed=claimed, read_connectives=frozenset(read)) == unclaimed


class _RecordingScope:
    """A cache scope that misses and records the `separate` marks its lookup keyed on."""

    def __init__(self) -> None:
        self.separate: list[bool] | None = None

    async def lookup_parsed(self, **kwargs):
        self.separate = list(kwargs["separate"])
        return None

    def note_health(self, *, timed_out: bool, degraded_leg: bool) -> None:
        pass


async def _handle(query: str, variables: list[str], scope: _RecordingScope) -> list[bool]:
    extraction = Extraction(kind="resolution", variables=variables)
    combined: list[bool] = []

    async def _extract(query, *, llm):
        return ExtractionResult(extraction=extraction, qualifiers={}), None

    async def _resolve_variable(variable, **kwargs):
        return _stub_region(variable)

    async def _combiner(regions, **kwargs):
        combined.extend(region.separate for region in regions)
        yield None

    graph = FakeGraph()
    with (
        patch("search.engine.entry.raw_text.extract", _extract),
        patch("search.engine.fanout.resolve_variable", _resolve_variable),
        patch("search.engine.entry.raw_text._stream_combined_regions", _combiner),
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
                scope=scope,  # ty: ignore[invalid-argument-type]
            ):
                pass
    return combined


@pytest.mark.parametrize(
    ("query", "variables"),
    [
        (_COMPARE, ["health aid", "education aid"]),
        ("compare health ODA with basic health ODA", ["health ODA", "basic health ODA"]),
        ("health ODA vs. basic health ODA", ["basic health ODA", "health ODA"]),
    ],
)
def test_the_raw_door_carries_the_marks_to_the_legs_and_the_cache_lookup(caplog, query, variables):
    scope = _RecordingScope()
    caplog.set_level(logging.INFO, logger=raw_text.__name__)

    legs = asyncio.run(_handle(query, variables, scope))

    assert legs == [True, True]
    assert scope.separate == [True, True]
    assert not [r for r in caplog.records if r.getMessage().startswith("raw_query_coverage")]


def test_the_raw_door_logs_the_words_no_part_claims(caplog):
    caplog.set_level(logging.INFO, logger=raw_text.__name__)

    asyncio.run(_handle("maternal health aid and debt", ["health aid", "debt"], _RecordingScope()))

    assert [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("raw_query_coverage")
    ] == ["raw_query_coverage unclaimed=maternal"]


def test_a_marked_variable_keys_layer2_apart_from_the_unmarked_extraction():
    async def _key(separate: list[bool]) -> str | None:
        cache = ResponseCache(max_bytes=10_000_000, max_entry_bytes=10_000_000, ttl_s=60)
        scope = _make_scope(cache)
        await scope.lookup_parsed(
            variables=["health aid", "education aid"],
            entities=[],
            dates=[],
            dropped_qualifiers=[],
            variable_qualifiers=[(), ()],
            operations=[(), ()],
            query=_COMPARE,
            entity_roles={},
            sources=[(), ()],
            query_text=None,
            separate=separate,
        )
        await scope.release()
        return scope._layer2_key

    marked, unmarked = asyncio.run(_key([True, True])), asyncio.run(_key([False, False]))

    assert marked is not None
    assert marked != unmarked
