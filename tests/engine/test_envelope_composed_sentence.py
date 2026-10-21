"""Pins composed sentence fields and text across entry paths and HTTP transports.

Issue #598 covers unresolved or disagreeing parts. Issue #593 covers two definite parts with
matching qualifiers, exercised through dataset resubmit because no raw-text fixture reaches that case.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from search.engine.assemble import build_no_data_part
from search.engine.core import _quick_no_data
from search.engine.envelope import make_diagnostics, make_response, now_ms
from search.engine.extract import Extraction, ExtractionResult
from search.models import QueryEcho, ResubmitIdentityEntry, Slot
from tests.engine._axis_doubles import TOTAL_POPULATION_INDIA_FAMILY, axis_confirming_reader
from tests.engine._base_identity import (
    GDP_SHAPE,
    make_identity,
    subject_entity,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    make_request,
    member_walk_reader,
    offline_resolve,
    with_member_arcs,
)
from tests.engine._search_app import search_app
from tests.engine.test_parts_shape import (
    _MIDDLE_FAILS_QUERY,
    _MIDDLE_FAILS_VARIABLES,
    _llm_with_extraction,
)
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def _client(*, walk: bool = False) -> TestClient:
    """A Search app over the shared fixture graph; with `walk`, a resubmit walks the mint's
    member enumeration over its nodes."""
    graph = FakeGraph()
    return TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=member_walk_reader(graph) if walk else None,
        ),
        raise_server_exceptions=True,
    )


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Return SSE event names and decoded data, skipping ping comments."""
    frames: list[tuple[str, dict]] = []
    event_type: str | None = None
    for line in body.splitlines():
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:") and event_type is not None:
            frames.append((event_type, json.loads(line[len("data:") :].strip())))
            event_type = None
    return frames


def _assert_envelope_sentence(client: TestClient, request: dict) -> None:
    """Assert the sentence fields are present in buffered and SSE responses."""
    buffered = client.post("/api/search/resolve", json=request)
    assert buffered.status_code == 200
    sse = client.post("/api/search/resolve", json=request, headers={"Accept": "text/event-stream"})
    assert sse.status_code == 200
    buffered_body = buffered.json()
    sse_response = next(d for ev, d in _parse_sse(sse.text) if ev == "response")["response"]
    for body in (buffered_body, sse_response):
        assert body.get("rendered_sentence")
        assert body.get("sentence_plan")


# One test per ResolveResponse construction site funnelling through
# make_response: raw_text N=1, raw_text N>=2, resubmit, and the three
# single-part no-data shortcuts.


def test_raw_text_n1_envelope_sentence():
    """conjoin.py's assemble_region, reached by the raw_text N=1 path."""
    client = _client()
    request = {"input": {"kind": "raw_text", "query": "total population India"}}
    _assert_envelope_sentence(client, request)


def test_raw_text_n_ge_2_envelope_sentence():
    """conjoin.py's combine_regions, reached by the raw_text N>=2 path."""
    client = _client()
    request = {"input": {"kind": "raw_text", "query": "population and GDP in Brazil"}}
    _assert_envelope_sentence(client, request)


def _gdp_resubmit_request() -> dict:
    """Build a definite GDP resubmit request."""
    entities = [subject_entity("country/IND")]
    return {
        "input": {
            "kind": "dataset_resubmit",
            "entries": [
                {
                    "entry_kind": "identity",
                    "variable_text": "a resubmitted identity",
                    "identity": make_identity(
                        GDP_SHAPE,
                        [
                            Slot.model_validate(
                                {
                                    "key": {
                                        "axis": "how",
                                        "property": {
                                            "ref": "activitySource",
                                            "label": "activity source",
                                        },
                                        "label": "activity source",
                                    },
                                    "binding": {"kind": "unbound"},
                                }
                            )
                        ],
                        entities=entities,
                    ).model_dump(mode="json"),
                }
            ],
        },
        "response_options": {},
    }


def test_resubmit_envelope_sentence():
    """Checks the resubmit sentence path."""
    client = _client(walk=True)
    resp = client.post("/api/search/resolve", json=_gdp_resubmit_request())
    assert resp.status_code == 200
    assert resp.json()["status"] == "definite"
    _assert_envelope_sentence(client, _gdp_resubmit_request())


def test_empty_query_no_data_envelope_sentence():
    """Whitespace queries carry the envelope sentence."""
    client = _client()
    request = {"input": {"kind": "raw_text", "query": "   "}}
    _assert_envelope_sentence(client, request)


def test_no_variables_extracted_no_data_envelope_sentence(monkeypatch):
    """An empty extraction carries the envelope sentence."""

    async def _empty_extraction(query: str, *, llm):
        return (
            ExtractionResult(
                extraction=Extraction(kind="resolution", variables=[], entities=[], dates=[]),
                qualifiers={},
            ),
            None,
        )

    monkeypatch.setattr("search.engine.entry.raw_text.extract", _empty_extraction)
    client = _client()
    request = {"input": {"kind": "raw_text", "query": "asdf qwerty"}}
    _assert_envelope_sentence(client, request)


def test_quick_no_data_shortcut_envelope_sentence():
    """The unreachable `_quick_no_data` fallback carries the envelope sentence."""
    response = _quick_no_data(
        reason="variable_not_resolved", query="", engine_build="test", start_ms=now_ms()
    )
    assert response.rendered_sentence
    assert response.sentence_plan


def test_single_part_no_data_still_names_variable_in_full_standalone_sentence():
    """A single no-data part mirrors its own sentence, retaining the full variable name."""
    part = build_no_data_part(
        "variable_not_resolved",
        variable_text="unemployment rate",
        warnings=[],
    )
    assert part.rendered_sentence == (
        'Could not match "unemployment rate" to a supported indicator. Try a different wording '
        "or a related indicator."
    )
    response = make_response(
        QueryEcho(
            entry_path="raw_text",
            raw_query="unemployment rate",
            normalized_query=None,
            extract_skipped=False,
        ),
        make_diagnostics("test", [], {}, 0),
        [part],
    )
    assert response.rendered_sentence == part.rendered_sentence
    assert response.rendered_sentence.lower().count("unemployment rate") == 1


def test_raw_text_three_variable_composed_sentence_names_middle_variable_once():
    """Reuses test_parts_shape.py's own "population and left-handedness and
    birth rate in Ethiopia" fixture (test_three_variable_query_middle_fails):
    population (index 0) resolves definite, left-handedness (index 1) is
    genuinely unresolved (variable_not_resolved, one of
    composed_variable_outcome's five reasons), and birth rate (index 2) lands on
    candidates. The composed sentence names an unresolved variable exactly
    once: its own slot token carries the bare variable_text, framed by the
    reason's own (prefix, suffix) pair, which states the outcome without
    repeating the term. The variable appears once.
    The sibling CandidatesPart names its variable once too, as the slot before its
    count of readings."""
    fake_llm = _llm_with_extraction(
        _MIDDLE_FAILS_QUERY, variables=_MIDDLE_FAILS_VARIABLES, entities=["Ethiopia"]
    )
    result = offline_resolve(make_request(_MIDDLE_FAILS_QUERY), llm=fake_llm)
    assert result.rendered_sentence is not None
    assert result.rendered_sentence.lower().count("left-handedness") == 1
    assert "Birth rate has 2 readings." in result.rendered_sentence


# Goldens: "population and birth rate in Ethiopia", the disagree/unresolved case

_ETHIOPIA_REQUEST = {
    "input": {"kind": "raw_text", "query": "population and birth rate in Ethiopia"},
    "response_options": {},
}

_ETHIOPIA_RENDERED_SENTENCE = (
    "Answered 1 of 2 parts. "
    "Ethiopia's total population, 1960 to 2024. World Development Indicators. "
    "Birth rate has 2 readings. Ethiopia's crude birth rate per capita. 1960 to 2023. "
    "Ethiopia's female fertility rate. World Development Indicators \u00b7 1960 to 2023."
)


def test_ethiopia_population_and_birth_rate_composed_sentence_buffered():
    """Issue #598: an unresolved sibling keeps both parts' sentences separate and unhoisted."""
    client = _client()
    resp = client.post("/api/search/resolve", json=_ETHIOPIA_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "mixed"
    assert [p["status"] for p in body["parts"]] == ["definite", "candidates"]
    assert body["rendered_sentence"] == _ETHIOPIA_RENDERED_SENTENCE
    assert "(" not in body["rendered_sentence"] and ")" not in body["rendered_sentence"]

    plan = body["sentence_plan"]
    population_head = next(
        t for t in plan if t["kind"] == "slot" and t["text"] == "total population"
    )
    assert population_head["part_index"] == 0
    birth_rate_slot = next(t for t in plan if t["kind"] == "slot" and t["text"] == "Birth rate")
    assert birth_rate_slot["part_index"] == 1
    where_slot = next(t for t in plan if t.get("axis") == "where")
    assert where_slot["text"] == "Ethiopia"
    assert where_slot["part_index"] == 0  # per-part: no hoisting across an unresolved sibling
    when_slot = next(t for t in plan if t.get("axis") == "when")
    assert when_slot["text"] == "1960 to 2024"
    assert when_slot["part_index"] == 0

    # The mixed answer opens with two unindexed tokens, the opener and its space. After
    # them every token belongs to exactly one sentence (connectives and the terminal "."
    # included), joined by exactly one shared, unindexed " ".
    opener, body_plan = plan[:2], plan[2:]
    assert [(t["text"], t.get("part_index")) for t in opener] == [
        ("Answered 1 of 2 parts.", None),
        (" ", None),
    ]
    separators = [
        t
        for t in body_plan
        if t["kind"] == "text" and t["text"] == " " and t.get("part_index") is None
    ]
    assert len(separators) == 1
    assert {t.get("part_index") for t in body_plan if t is not separators[0]} == {0, 1}


def test_ethiopia_population_and_birth_rate_composed_sentence_sse():
    """The SSE response event carries the same composed sentence as the buffered body."""
    client = _client()
    resp = client.post(
        "/api/search/resolve",
        json=_ETHIOPIA_REQUEST,
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    response = next(d for ev, d in _parse_sse(resp.text) if ev == "response")["response"]
    assert response["rendered_sentence"] == _ETHIOPIA_RENDERED_SENTENCE


# Golden: "population and fertility rate in Brazil" -- both parts definite,
# "where" happens to agree while "when"/"source" do not

_FERTILITY_BRAZIL_REQUEST = {
    "input": {"kind": "raw_text", "query": "population and fertility rate in Brazil"},
    "response_options": {},
}

_FERTILITY_BRAZIL_RENDERED_SENTENCE = (
    "Brazil's total population, 1960 to 2024. Brazil's female fertility rate, "
    "1960 to 2023. World Development Indicators."
)


def test_brazil_population_and_fertility_rate_composed_sentence_disagree():
    """Issue #598: a disagreement on `when` or `source` prevents hoisting even the shared `where`."""
    client = _client()
    resp = client.post("/api/search/resolve", json=_FERTILITY_BRAZIL_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert [p["status"] for p in body["parts"]] == ["definite", "definite"]
    assert body["rendered_sentence"] == _FERTILITY_BRAZIL_RENDERED_SENTENCE
    assert "(" not in body["rendered_sentence"] and ")" not in body["rendered_sentence"]

    plan = body["sentence_plan"]
    where_tokens = [t for t in plan if t.get("axis") == "where"]
    assert len(where_tokens) == 2  # each part keeps its own, despite agreeing on text
    assert {t["text"] for t in where_tokens} == {"Brazil"}
    assert {t["part_index"] for t in where_tokens} == {0, 1}

    periods = [
        t for t in plan if t["kind"] == "text" and t["text"] == "." and t["segment"] == "heading"
    ]
    assert len(periods) == 2  # each heading grows its own terminal period
    assert {t["part_index"] for t in periods} == {0, 1}

    separators = [
        t for t in plan if t["kind"] == "text" and t["text"] == " " and t.get("part_index") is None
    ]
    assert len(separators) == 1
    assert {t.get("part_index") for t in plan if t is not separators[0]} == {0, 1}


# Golden: two definite parts whose qualifier axes fully agree -- the
# hoisting branch of the binary rule, reached through dataset_resubmit since
# no raw_text fixture in this bank produces two AGREEING definite parts (see
# this module's own docstring on why).


def _india_gdp_identity_entry() -> ResubmitIdentityEntry:
    """Build a definite India GDP identity with an unbound `activitySource` slot."""
    entities = [subject_entity("country/IND")]
    identity = make_identity(
        GDP_SHAPE,
        [
            Slot.model_validate(
                {
                    "key": {
                        "axis": "how",
                        "property": {"ref": "activitySource", "label": "activity source"},
                        "label": "activity source",
                    },
                    "binding": {"kind": "unbound"},
                }
            )
        ],
        entities=entities,
    )
    return ResubmitIdentityEntry(
        variable_text="GDP",
        identity=identity,
    )


def test_two_agreeing_definite_parts_hoist_via_dataset_resubmit():
    """Issue #593: two definite parts with matching qualifiers hoist once through dataset resubmit."""
    population = offline_resolve(make_request("total population India")).parts[0]
    assert population.status == "definite"

    entries = [
        ResubmitIdentityEntry(
            variable_text="population",
            identity=population.spec.definition.identity,
        ),
        _india_gdp_identity_entry(),
    ]
    request = {
        "input": {
            "kind": "dataset_resubmit",
            "entries": [e.model_dump(mode="json") for e in entries],
        },
        "response_options": {},
    }
    graph = FakeGraph()
    client = TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=with_member_arcs(
                axis_confirming_reader(*TOTAL_POPULATION_INDIA_FAMILY), graph
            ),
        ),
        raise_server_exceptions=True,
    )
    buffered = client.post("/api/search/resolve", json=request)
    assert buffered.status_code == 200
    sse = client.post("/api/search/resolve", json=request, headers={"Accept": "text/event-stream"})
    assert sse.status_code == 200
    sse_response = next(d for ev, d in _parse_sse(sse.text) if ev == "response")["response"]

    for body in (buffered.json(), sse_response):
        assert body["status"] == "definite"
        assert [p["status"] for p in body["parts"]] == ["definite", "definite"]
        rendered = body["rendered_sentence"]
        assert "India" in rendered
        # The unit phrase's price basis is the only parenthetical. A part never attaches one.
        unattached = rendered.replace(" (current prices)", "")
        assert "(" not in unattached and ")" not in unattached
        plan = body["sentence_plan"]
        where_tokens = [t for t in plan if t.get("axis") == "where"]
        when_tokens = [t for t in plan if t.get("axis") == "when"]
        source_tokens = [t for t in plan if t.get("axis") == "source"]
        # The shared possessive place lead opens the sentence once; when and source
        # hoist too.
        assert [t["text"] for t in where_tokens] == ["India"]
        assert len(when_tokens) == 1
        assert len(source_tokens) == 1
        # Hoisted: the wire omits a None part_index entirely (see the
        # sibling goldens above, which pin the disagreeing/unresolved
        # branches' explicit part_index instead).
        for token in (*where_tokens, *when_tokens, *source_tokens):
            assert "part_index" not in token
