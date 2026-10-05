"""The eval runner against the Search app in process, over httpx's ASGI transport."""

from __future__ import annotations

import httpx

from search import RESOLVE_RESPONSE_ADAPTER
from search.eval.cases import validate_case
from search.eval.runner import CaseRun, fetch_schema, run_cases
from search.models import SCHEMA_VERSION
from tests.engine._harness import EMPTY_SDMX, run_pinned
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_URL = "http://search"

# Definite, candidates and no-data answers on the raw-text door, plus the parsed door.
_CASES = [
    {"id": "bd-01", "query": "fertility rate in Kenya"},
    {"id": "bd-02", "query": "life expectancy in Kenya"},
    {"id": "bd-03", "query": "population and GDP in Brazil"},
    {"id": "bd-04", "query": "total population India"},
    {"id": "bd-05", "query": "GDP of Brazil"},
    {"id": "bd-06", "query": "under-5 child mortality rate"},
    {"id": "nd-01", "query": "teacher count in Nauru"},
    {"id": "nd-02", "query": "left-handedness rate in France"},
    {
        "id": "pd-01",
        "query": "fertility rate in Kenya",
        "door": "parsed",
        "parsed": [{"variable_text": "fertility rate", "entities": [{"text": "Kenya"}]}],
    },
]


def _run() -> tuple[dict, dict[str, CaseRun]]:
    graph = FakeGraph()
    app = search_app(graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=EMPTY_SDMX)
    cases = [
        validate_case({**raw, "why": "Shape only.", "expect": {"definite": {}}}, families={})
        for raw in _CASES
    ]

    async def go() -> tuple[dict, dict[str, CaseRun]]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            schema = await fetch_schema(_URL, client=client)
            runs = await run_cases(cases, url=_URL, samples=1, concurrency=4, client=client)
        return schema, runs

    return run_pinned(go())


def test_every_case_returns_a_clean_response_that_validates_against_the_models():
    schema, runs = _run()

    assert schema["schema_version"] == SCHEMA_VERSION
    assert "properties" in schema["response"] or "oneOf" in schema["response"]
    assert list(runs) == [raw["id"] for raw in _CASES]
    for case_id, run in runs.items():
        assert run.infra_error is None, case_id
        assert run.modal is not None
        assert (run.modal.status, run.modal.degraded, run.modal.request_rejected) == (
            200,
            False,
            False,
        ), case_id
        RESOLVE_RESPONSE_ADAPTER.validate_python(run.modal.body)
        assert run.modal.body["schema_version"] == SCHEMA_VERSION
