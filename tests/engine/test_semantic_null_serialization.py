"""Serialization tests distinguish meaningful nulls from absent optional fields.

Blank observations are dropped. Unconfirmed option coverage stays null on SSE and is omitted
from buffered output when unrequested; assertions inspect the serialized payloads.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import GraphReader
from search.engine.events import EntityOptionsEvent, SlotOptionsEvent, serialize_sse
from search.models import (
    EntityOptions,
    SlotKey,
    SlotOption,
    SlotValue,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver

# A single missing-value point ("2022" has no "value" key) alongside two real
# ones -- reused from test_observations_e2e_buffered_sse.py's _IND_OBS fixture.
_IND_OBS = {
    "Count_Person|country/IND": [
        {
            "earliestDate": "2020",
            "latestDate": "2022",
            "obsCount": 3,
            "provenanceId": "dc/base/WorldPopulationProspects",
            "observations": [
                {"date": "2020", "value": 100.0},
                {"date": "2021", "value": 101.0},
                {"date": "2022"},
            ],
        }
    ],
}


def _ind_client() -> TestClient:
    """A client wired with real materialise observations over _IND_OBS."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_IND_OBS)
    app = search_app(
        graph=FakeGraph(obs=_IND_OBS),
        resolver=make_resolver(FakeGraph(obs=_IND_OBS)),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    return TestClient(app, raise_server_exceptions=False)


_IND_REQUEST = {
    "input": {"kind": "raw_text", "query": "total population India"},
    "response_options": {"include_observations": True},
}

# FertilityRate_Person_Female's gender slot names one value (Female): the
# taxonomy this shape's own confirmed member group carries. include_option_
# coverage still probes it, so this is the case that distinguishes "never
# probed" (buffered, omitted) from "probed, unconfirmed" (SSE, honest null).
_FERTILITY_REQUEST = {"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}}


def _parse_sse(body: str) -> list[tuple[str, dict]]:
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


def _client(**graph_kwargs) -> TestClient:
    return _client_with_graph(FakeGraph(**graph_kwargs))


def _client_with_graph(
    graph: FakeGraph,
    *,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
) -> TestClient:
    """Build a client with injected materialise readers to avoid live graph calls."""
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    return TestClient(app, raise_server_exceptions=False)


def _gender_options(spec_dict: dict) -> list[dict]:
    return next(
        so
        for so in spec_dict["resolved"]["slot_options"]
        if (so["key"].get("property") or {}).get("ref") == "gender"
    )["options"]


# ---------------------------------------------------------------------------
# ObservationPoint.value -- unconditional (buffered and SSE alike)
# ---------------------------------------------------------------------------


def test_observation_point_value_null_survives_buffered_body():
    """Materialise drops the blank 2022 point; the 2020 and 2021 values remain."""
    client = _ind_client()
    resp = client.post("/api/search/resolve", json=_IND_REQUEST)
    assert resp.status_code == 200
    points = resp.json()["parts"][0]["spec"]["resolved"]["observations"]["series"][0]["points"]
    assert [p["date"] for p in points] == ["2020", "2021"]
    assert [p["value"] for p in points] == [100.0, 101.0]


def test_observation_point_value_null_survives_sse_trailing_event():
    """SSE also omits the blank 2022 point."""
    client = _ind_client()
    resp = client.post(
        "/api/search/resolve",
        json=_IND_REQUEST,
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    obs_event = next(d for ev, d in frames if ev == "observations")
    points = obs_event["observations"]["series"][0]["points"]
    assert [p["date"] for p in points] == ["2020", "2021"]
    assert [p["value"] for p in points] == [100.0, 101.0]


# ---------------------------------------------------------------------------
# SlotOption.coverage -- context-gated: null survives on SSE, omitted on
# buffered when never probed
# ---------------------------------------------------------------------------


_FERTILITY_KEN_OBS_KEY = "FertilityRate_Person_Female|country/KEN"
"""The one fixture-backed (stat_var, entity) pair the option-coverage probe has
to answer over for the fertility query below: its gender taxonomy names
FertilityRate_Person_Female, so the probe's sibling-member set is that one
StatVar over the resolved place -- the same graph_obs.json facts
test_option_coverage.py's own materialise-backed probe reads."""


def _fertility_probe_double(*, fail_probe: bool):
    """Build the fertility probe fixture, optionally failing its batched read."""
    return materialise_double_from_obs(
        {_FERTILITY_KEN_OBS_KEY: _load_json("graph_obs.json")[_FERTILITY_KEN_OBS_KEY]},
        fail_option_coverage_probe=fail_probe,
    )


def _fertility_sse_frames(fail_probe: bool) -> list[tuple[str, dict]]:
    """Parse the fertility query's SSE stream, with the option-coverage probe
    requested and its own materialise pair wired to `fail_probe`."""
    graph = FakeGraph()
    materialise_graph, materialise_sdmx = _fertility_probe_double(fail_probe=fail_probe)
    client = _client_with_graph(
        graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    resp = client.post(
        "/api/search/resolve",
        json={
            **_FERTILITY_REQUEST,
            "response_options": {"include_option_coverage": True},
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    return _parse_sse(resp.text)


def _gender_slot_options_event(frames: list[tuple[str, dict]]) -> dict:
    return next(
        d
        for ev, d in frames
        if ev == "slot_options" and (d["key"].get("property") or {}).get("ref") == "gender"
    )


def _done_frame(frames: list[tuple[str, dict]]) -> dict:
    return next(d for ev, d in frames if ev == "done")


def _probe_failure_warning(frames: list[tuple[str, dict]]) -> dict:
    """Find the probe-failure warning on the terminal `done` event."""
    return next(
        w for w in _done_frame(frames)["warnings"] if w["code"] == "option_coverage_probe_failed"
    )


def test_slot_option_coverage_null_survives_sse_trailing_event():
    """A failed batch leaves every gender option at null coverage and adds a warning to `done`.

    The main resolve remains definite; a successful empty probe would return zero-count coverage.
    """
    frames = _fertility_sse_frames(fail_probe=True)
    response = next(d for ev, d in frames if ev == "response")["response"]
    assert response["status"] == "definite"
    options = _gender_slot_options_event(frames)["options"]
    assert options, "sanity: the gender taxonomy is non-empty"
    assert all("coverage" in o and o["coverage"] is None for o in options)
    warning = _probe_failure_warning(frames)
    assert warning["severity"] == "warn"
    assert warning["spec_id"] == response["parts"][0]["spec"]["spec_id"]


def test_slot_option_coverage_probe_reaches_the_graph():
    """A successful probe returns coverage for each gender option without a failure warning."""
    frames = _fertility_sse_frames(fail_probe=False)
    options = _gender_slot_options_event(frames)["options"]
    assert options, "sanity: the gender taxonomy is non-empty"
    assert all(o["coverage"] is not None for o in options)
    assert _done_frame(frames)["warnings"] == []


def test_slot_option_coverage_omitted_on_buffered_when_never_probed():
    """Without include_option_coverage, the buffered body's gender options
    omit the `coverage` key entirely -- emitting a null there would read as
    "we probed and found nothing", which is false, so the item shape stays
    {ref, label}-only."""
    client = _client()
    resp = client.post("/api/search/resolve", json=_FERTILITY_REQUEST)
    assert resp.status_code == 200
    spec = resp.json()["parts"][0]["spec"]
    options = _gender_options(spec)
    assert options  # sanity: the taxonomy is non-empty
    assert not any("coverage" in o for o in options)


def test_entity_options_coverage_null_survives_sse_trailing_event():
    """Entity options reuse `SlotOption`; direct serialization covers an unconfirmed donor value."""
    unconfirmed = SlotOption(value=SlotValue(value_kind="entity"), coverage=None)
    confirmed = SlotOption(value=SlotValue(value_kind="entity"))
    ev = EntityOptionsEvent(
        spec_id="s1",
        entity_ref="country/FRA",
        role="origin",
        options=[unconfirmed, confirmed],
        truncated=False,
    )
    frame = serialize_sse(ev)
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["options"][0]["coverage"] is None
    assert "coverage" in payload["options"][0]


def test_entity_options_coverage_omitted_on_buffered_when_never_probed():
    """Direct serialization covers omission because the fixture has no unconfirmed donor option."""
    eo = EntityOptions(
        entity_ref="country/FRA",
        role="origin",
        options=[SlotOption(value=SlotValue(value_kind="entity"))],
        truncated=False,
        source="observed",
    )
    dumped = eo.model_dump(mode="json", exclude_none=True)
    assert dumped["options"]  # sanity: candidates exist
    assert not any("coverage" in o for o in dumped["options"])


def test_slot_options_event_coverage_null_survives_direct_serialization():
    """Direct, minimal check of the model/event boundary (no pipeline
    involved): a SlotOptionsEvent carrying an unconfirmed option must keep
    coverage: null through serialize_sse's exclude_none."""
    opt = SlotOption(value=SlotValue(value_kind="literal", literal="foo"), coverage=None)
    key = SlotKey(axis="what", label="purpose")
    ev = SlotOptionsEvent(spec_id="s1", key=key, options=[opt], truncated=False)
    frame = serialize_sse(ev)
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["options"][0]["coverage"] is None


def test_slot_option_coverage_omitted_without_sse_context():
    opt = SlotOption(value=SlotValue(value_kind="literal", literal="foo"), coverage=None)
    assert (
        opt.model_dump_json(exclude_none=True)
        == '{"value":{"value_kind":"literal","literal":"foo"}}'
    )


# ---------------------------------------------------------------------------
# Regression guard: a genuinely absent optional member (no meaning in null)
# stays omitted on both boundaries, independent of the semantic-null handling
# for the two fields above.
# ---------------------------------------------------------------------------


def test_genuinely_absent_optional_stays_omitted_on_both_boundaries():
    """With skills off, `skills` stays absent from buffered and SSE responses."""
    client = _client()
    request = {**_FERTILITY_REQUEST, "response_options": {}}
    buffered = client.post("/api/search/resolve", json=request)
    sse = client.post(
        "/api/search/resolve",
        json=request,
        headers={"Accept": "text/event-stream"},
    )
    assert buffered.status_code == 200
    assert sse.status_code == 200
    buffered_body = buffered.json()
    assert "skills" not in buffered_body

    sse_response = next(d for ev, d in _parse_sse(sse.text) if ev == "response")["response"]
    assert "skills" not in sse_response


def test_sentence_token_distinguishes_omitted_when_not_marked():
    """Definite part-level tokens omit `distinguishes` and `part_index` on both transports."""
    client = _client()
    request = {**_FERTILITY_REQUEST, "response_options": {}}
    buffered = client.post("/api/search/resolve", json=request)
    sse = client.post(
        "/api/search/resolve",
        json=request,
        headers={"Accept": "text/event-stream"},
    )
    assert buffered.status_code == 200
    assert sse.status_code == 200

    buffered_body = buffered.json()
    assert buffered_body["status"] == "definite"
    buffered_plan = buffered_body["parts"][0]["spec"]["resolved"]["sentence_plan"]
    assert buffered_plan  # sanity: tokens exist
    for token in buffered_plan:
        assert "distinguishes" not in token
        assert "part_index" not in token

    sse_response = next(d for ev, d in _parse_sse(sse.text) if ev == "response")["response"]
    sse_plan = sse_response["parts"][0]["spec"]["resolved"]["sentence_plan"]
    assert sse_plan
    for token in sse_plan:
        assert "distinguishes" not in token
        assert "part_index" not in token


# ---------------------------------------------------------------------------
# SentenceToken.kind=="note" -- a required field, never null-omitted, but
# the trailing window-note token render_sentence_plan appends after the
# plan's own terminal "." token is worth pinning independently on both
# transports.
# ---------------------------------------------------------------------------

_FERTILITY_KENYA_1950_TO_1965_REQUEST = {
    "input": {
        "kind": "parsed",
        "parts": [
            {
                "variable_text": "fertility rate",
                "entities": [{"text": "Kenya"}],
                "time_text": "1950 to 1965",
            }
        ],
    },
    "response_options": {},
}


def test_sentence_token_segments_survive_both_boundaries():
    """ "1950 to 1965" starts before this series' 1960-2023 range, so the answer serves 1960
    to 1965; both transports carry the same plan, its period in the heading and its source
    in the byline."""
    client = _client()
    buffered = client.post("/api/search/resolve", json=_FERTILITY_KENYA_1950_TO_1965_REQUEST)
    sse = client.post(
        "/api/search/resolve",
        json=_FERTILITY_KENYA_1950_TO_1965_REQUEST,
        headers={"Accept": "text/event-stream"},
    )
    assert buffered.status_code == 200
    assert sse.status_code == 200

    buffered_body = buffered.json()
    assert buffered_body["status"] == "definite"
    buffered_plan = buffered_body["parts"][0]["spec"]["resolved"]["sentence_plan"]
    when = next(token for token in buffered_plan if token.get("axis") == "when")
    assert (when["text"], when["segment"]) == ("1960 to 1965", "heading")
    assert buffered_plan[-1]["segment"] == "byline"

    sse_response = next(d for ev, d in _parse_sse(sse.text) if ev == "response")["response"]
    sse_plan = sse_response["parts"][0]["spec"]["resolved"]["sentence_plan"]
    assert sse_plan == buffered_plan


@pytest.mark.parametrize(
    ("time_text", "asked"),
    [("1875", "1875"), ("since 2025", "2025 to 2026")],
)
def test_a_window_without_points_answers_no_data_naming_the_span_the_series_runs(
    time_text: str, asked: str
):
    """Kenya's fertility series runs 1960 to 2023, so "1875" and "since 2025" (read to the
    pinned 2026) hold no point: both transports answer `no_observations` for the window,
    and the part departs as `year_unavailable` for the years read instead of claiming Kenya
    has no data."""
    request = {
        "input": {
            "kind": "parsed",
            "parts": [
                {
                    "variable_text": "fertility rate",
                    "entities": [{"text": "Kenya"}],
                    "time_text": time_text,
                }
            ],
        },
        "response_options": {},
    }
    client = _client()
    buffered = client.post("/api/search/resolve", json=request)
    sse = client.post("/api/search/resolve", json=request, headers={"Accept": "text/event-stream"})
    assert buffered.status_code == sse.status_code == 200
    sse_response = next(d for ev, d in _parse_sse(sse.text) if ev == "response")["response"]

    for body in (buffered.json(), sse_response):
        part = body["parts"][0]
        assert part["status"] == "no_data"
        assert part["no_data"]["reason"] == "no_observations"
        assert part["no_data"].get("gate") is None
        assert [(d["code"], d["asked"]) for d in part["departures"]] == [
            ("year_unavailable", asked)
        ]
        assert "has no data for" not in part["rendered_sentence"]
