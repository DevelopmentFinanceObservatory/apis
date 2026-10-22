"""A relative date the raw-text extraction cannot read leaves the request dateless and
says so with `time_text_not_parsed`, as the parsed door does."""

from __future__ import annotations

from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT, Extraction
from search.engine.time_bind import TIME_TEXT_NOT_PARSED
from tests.engine._harness import make_request, offline_resolve
from tests.fixtures import FakeGraph, FakeLLM
from tests.fixtures.keys import fixture_key


def test_an_unread_relative_date_reaches_the_envelope_as_a_warning():
    query = "aid to Brazil over the last 0 years"
    llm = FakeLLM(
        responses={
            fixture_key(Extraction, _EXTRACTION_SYSTEM_PROMPT, query): {
                "kind": "resolution",
                "variables": ["aid"],
                "entities": [{"text": "Brazil", "span": "Brazil"}],
                "dates": [{"kind": "range", "relative": "over the last 0 years"}],
            }
        }
    )
    graph = FakeGraph(detect={}, resolve={"Brazil": "country/BRA"})
    result = offline_resolve(make_request(query), graph=graph, llm=llm)
    assert TIME_TEXT_NOT_PARSED in [w.code for w in result.diagnostics.warnings]
