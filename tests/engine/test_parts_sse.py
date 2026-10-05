"""Pins SSE event joins and fields absent from response payloads."""

from __future__ import annotations

import asyncio
from typing import Any

from search.engine.core import resolve_events
from search.engine.events import (
    ResponseEvent,
    SentenceEvent,
    ShapeEvent,
    StepEvent,
)
from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT, Extraction
from search.models import RawTextInput, ResolveOptions, ResolveRequest
from tests.engine._harness import (
    EMPTY_SDMX,
    PINNED_DATE,
    make_request,
    materialise_double_from_obs,
    offline_resolve,
)
from tests.fixtures import FakeGraph, FakeLLM, _load_json
from tests.fixtures.keys import fixture_key

_EXTRACT_SYS = _EXTRACTION_SYSTEM_PROMPT.replace("[[TODAY]]", PINNED_DATE.isoformat())


def _extract_key(query: str) -> str:
    return fixture_key(Extraction, _EXTRACT_SYS, query)


def _run(coro):
    return asyncio.run(coro)


async def _collect(gen):
    return [ev async for ev in gen]


# ---------------------------------------------------------------------------
# variable_index joins step/shape/sentence events onto the right response part
# ---------------------------------------------------------------------------

_THREE_VAR_QUERY = "population and left-handedness and GDP in Brazil"
_THREE_VARS = ["population", "left-handedness", "GDP"]


def _fake_llm_for_three_var_query() -> FakeLLM:
    return FakeLLM(
        responses={
            **_load_json("llm_responses.json"),
            _extract_key(_THREE_VAR_QUERY): {
                "kind": "resolution",
                "variables": _THREE_VARS,
                "entities": [{"text": "Brazil"}],
                "dates": [],
            },
        }
    )


def test_sse_variable_index_joins_step_shape_sentence_onto_response_parts():
    """Each per-variable event indexes the matching part in query order."""

    async def _body():
        events = await _collect(
            resolve_events(
                make_request(_THREE_VAR_QUERY),
                graph=FakeGraph(),
                llm=_fake_llm_for_three_var_query(),
                sdmx=EMPTY_SDMX,
            )
        )
        response_events = [e for e in events if isinstance(e, ResponseEvent)]
        assert len(response_events) == 1
        response = response_events[0].response
        assert [p.variable_text for p in response.parts] == _THREE_VARS

        per_variable_events = [
            e for e in events if isinstance(e, (ShapeEvent, SentenceEvent, StepEvent))
        ]
        checked_any = False
        for ev in per_variable_events:
            if ev.variable_index is None:
                continue
            checked_any = True
            assert response.parts[ev.variable_index].variable_text == _THREE_VARS[ev.variable_index]
        assert checked_any, "expected at least one per-variable event with a variable_index"

    _run(_body())


# ---------------------------------------------------------------------------
# Trailing per-spec events join on spec_id, not variable_index
# ---------------------------------------------------------------------------

_FERTILITY_KEY = "FertilityRate_Person_Female|country/BRA"


def test_sse_trailing_per_spec_events_join_on_spec_id():
    """Trailing slot options join the two definite parts by `spec_id`.

    The third variable has no spec, and each definite part emits an event.
    """
    query = "population and fertility rate and left-handedness in Brazil"
    variables = ["population", "fertility rate", "left-handedness"]
    fake_llm = FakeLLM(
        responses={
            **_load_json("llm_responses.json"),
            _extract_key(query): {
                "kind": "resolution",
                "variables": variables,
                "entities": [{"text": "Brazil"}],
                "dates": [],
            },
        }
    )
    obs = _load_json("graph_obs.json")
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {_FERTILITY_KEY: obs[_FERTILITY_KEY]}
    )

    async def _body():
        events = await _collect(
            resolve_events(
                ResolveRequest(
                    input=RawTextInput(query=query),
                    response_options=ResolveOptions(include_option_coverage=True),
                ),
                graph=FakeGraph(),
                llm=fake_llm,
                sdmx=EMPTY_SDMX,
                stream_slot_options=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            )
        )
        response_events = [e for e in events if isinstance(e, ResponseEvent)]
        assert len(response_events) == 1
        response = response_events[0].response
        assert [p.variable_text for p in response.parts] == variables

        population_part, fertility_part, left_handedness_part = response.parts
        assert population_part.status == "definite"
        assert fertility_part.status == "definite"
        assert left_handedness_part.status == "no_data"
        known_spec_ids = {population_part.spec.spec_id, fertility_part.spec.spec_id}

        slot_options_events = [e for e in events if e.kind == "slot_options"]
        assert slot_options_events, "expected at least one trailing slot_options event"
        assert all(e.spec_id in known_spec_ids for e in slot_options_events), (
            "every trailing slot_options event must join onto one of the two "
            "definite parts' own spec_id -- never the failed part, which "
            "mints no spec at all"
        )
        assert fertility_part.spec.spec_id in {e.spec_id for e in slot_options_events}, (
            "expected at least one trailing slot_options event to join onto "
            "the fertility rate part's own spec_id"
        )

    _run(_body())


# ---------------------------------------------------------------------------
# No response, buffered or streamed, carries a removed field anywhere
# ---------------------------------------------------------------------------

# Issue #567 retires `Warning.part_index`. `SentenceToken.part_index` remains valid.
# Check the key only inside warning objects because both models use that name.
_REMOVED_KEYS = ("interpretations",)
_REMOVED_KEYS_IN_WARNINGS = ("part_index",)


def _assert_no_removed_fields(obj: Any, *, in_warnings: bool = False) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert key not in _REMOVED_KEYS, f"removed field {key!r} present in wire payload"
            if in_warnings:
                assert key not in _REMOVED_KEYS_IN_WARNINGS, (
                    f"removed field {key!r} present in wire payload"
                )
            if key == "query_echo":
                assert isinstance(value, dict)
                assert "variable_text" not in value, (
                    "query_echo.variable_text was removed; the parts' own "
                    "variable_text values replace it"
                )
            _assert_no_removed_fields(value, in_warnings=key == "warnings")
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_removed_fields(item, in_warnings=in_warnings)


def test_buffered_response_carries_no_removed_fields():
    result = offline_resolve(make_request(_THREE_VAR_QUERY), llm=_fake_llm_for_three_var_query())
    payload = result.model_dump(mode="json")
    _assert_no_removed_fields(payload)


def test_sse_response_event_carries_no_removed_fields():
    async def _body():
        events = await _collect(
            resolve_events(
                make_request(_THREE_VAR_QUERY),
                graph=FakeGraph(),
                llm=_fake_llm_for_three_var_query(),
                sdmx=EMPTY_SDMX,
            )
        )
        response_events = [e for e in events if isinstance(e, ResponseEvent)]
        assert len(response_events) == 1
        payload = response_events[0].response.model_dump(mode="json")
        _assert_no_removed_fields(payload)

    _run(_body())
