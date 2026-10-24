"""The app's shared memo serves repeated queries while graph, SDMX, and observation work runs again.

A replay client uses recorded LLM responses with the app's real prompts.
"""

from __future__ import annotations

import copy
import logging
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from graph_generation import FakeGeneration
from search.engine.app import create_app
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs, parse_sse
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM, make_resolver

_QUERY = "total population India"
_OPTIONS = {"include_observations": True, "include_option_coverage": True}

# The same single-series override tests/engine/test_observations_e2e_buffered_sse.py
# uses: FakeGraph reads it for coverage, materialise_double_from_obs for the
# preview, so the two halves are computed independently over one set of facts.
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
                {"date": "2022", "value": 102.0},
            ],
        }
    ],
}


class _UsageMeta:
    prompt_token_count = 120
    candidates_token_count = 40
    cached_content_token_count = 0


class _Response:
    """The slice of a genai response `LLM._fetch_text` reads."""

    def __init__(self, *, text: str) -> None:
        self.text = text
        self.model_version = "gemini-3.5-flash-lite-007"
        self.usage_metadata = _UsageMeta()


class _ReplayingModels:
    """Replays `FakeLLM` fixtures keyed by the real prompt and records schema names."""

    def __init__(self) -> None:
        self._fixtures = FakeLLM()
        self.calls: list[str] = []

    async def generate_content(self, *, model, contents, config):
        schema = config.response_schema
        self.calls.append(schema.__name__)
        parsed, _usage = await self._fixtures.generate_structured(
            prompt=contents, system=config.system_instruction, schema=schema
        )
        return _Response(text=parsed.model_dump_json())


class _ReplayingClient:
    def __init__(self) -> None:
        self.models = _ReplayingModels()
        # Both client paths share call tracking and the no-usage override.
        self.aio = SimpleNamespace(models=self.models)


class _CountingGraph(FakeGraph):
    """Counts detection and coverage reads on `FakeGraph`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.detect_svs_calls = 0
        self.observation_facets_batch_calls = 0

    def detect_svs(self, query):
        self.detect_svs_calls += 1
        return super().detect_svs(query)

    def variable_facets_batch(self, *args, **kwargs):
        self.observation_facets_batch_calls += 1
        return super().variable_facets_batch(*args, **kwargs)


class _LogCapture(logging.Handler):
    """Captures `search` records because they do not propagate to `caplog`'s root handler."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record) -> None:
        self.messages.append(record.getMessage())


def _build():
    """Build the default app with a shared memo and call-counting fakes."""
    graph = _CountingGraph(obs=_IND_OBS)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_IND_OBS)
    sdmx = FakeSdmxClient(observations={}, availability={})
    client = _ReplayingClient()
    app = create_app(
        graph=graph,
        resolver=make_resolver(graph),
        sdmx=sdmx,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    return app, client, graph, materialise_graph, sdmx


def _resolve(client, *, sse: bool):
    headers = {"Accept": "text/event-stream"} if sse else {}
    return client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": _QUERY}, "response_options": _OPTIONS},
        headers=headers,
    )


@pytest.fixture
def wired(monkeypatch):
    """Wire the real app fallback to a replay client, avoiding credentials and network access.

    Patching `warm` and `_get_client` preserves the shared `LLM(memo=...)` path."""
    import search.engine.app as app_module
    import search.engine.clients.llm as llm_module

    capture = _LogCapture()
    search_logger = logging.getLogger("search")
    # An existing handler skips the app's setLevel call, so set INFO for capture.
    previous_level = search_logger.level
    search_logger.setLevel(logging.INFO)
    search_logger.addHandler(capture)
    app, replay_client, graph, materialise_graph, sdmx = _build()
    monkeypatch.setattr(llm_module, "_get_client", lambda: replay_client)
    monkeypatch.setattr(llm_module, "warm", AsyncMock())
    # The app's own LLM symbol, restored to the real class.
    monkeypatch.setattr(app_module, "LLM", llm_module.LLM)
    try:
        with patch("search.engine.extract.date") as mock_date:
            mock_date.today.return_value = PINNED_DATE
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            with TestClient(app, raise_server_exceptions=False) as test_client:
                yield test_client, replay_client, graph, materialise_graph, sdmx, capture
    finally:
        search_logger.removeHandler(capture)
        search_logger.setLevel(previous_level)


def _body(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


def _without_timings(body: dict) -> dict:
    """Copy the first part without per-request trace and timing fields."""
    part = copy.deepcopy(body["parts"][0])

    def strip(node):
        if isinstance(node, dict):
            for key in ("timing_ms", "trace"):
                node.pop(key, None)
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    strip(part)
    return part


def _assert_observations(body: dict) -> None:
    part = body["parts"][0]
    assert part["status"] == "definite", part
    observations = part["spec"]["resolved"]["observations"]
    assert observations["total_obs"] == 3
    assert [p["value"] for p in observations["series"][0]["points"]] == [100.0, 101.0, 102.0]


def test_app_builds_a_shared_memoised_llm_and_a_repeat_costs_no_gemini_call(wired):
    client, replay_client, graph, materialise_graph, sdmx, capture = wired

    first = _body(_resolve(client, sse=False))
    calls_after_first = list(replay_client.models.calls)
    _assert_observations(first)

    # The lifespan's own log line, so the shared LLM is the one under test.
    assert any("LLM memo built" in m for m in capture.messages), capture.messages

    graph_probes_before = (
        graph.detect_svs_calls,
        graph.observation_facets_batch_calls,
        materialise_graph.calls["variable_facets_batch"],
        materialise_graph.calls["observation_facets_batch_many"],
        len(sdmx.series_calls),
    )

    second = _body(_resolve(client, sse=False))

    # One extract, one bind and one quantity read, on the first request only.
    assert calls_after_first == ["Extraction", "QuantityOutput", "_UniversalBindOutput"]
    assert replay_client.models.calls == calls_after_first
    assert "Extraction store hit" in capture.messages
    assert "LLM memo hit: schema=_UniversalBindOutput" in capture.messages
    assert "LLM memo hit: schema=QuantityOutput" in capture.messages

    # The repeat's own answer, with observations recomputed live: every graph
    # and SDMX probe the first request made runs again.
    _assert_observations(second)
    assert _without_timings(second) == _without_timings(first)
    graph_probes_after = (
        graph.detect_svs_calls,
        graph.observation_facets_batch_calls,
        materialise_graph.calls["variable_facets_batch"],
        materialise_graph.calls["observation_facets_batch_many"],
        len(sdmx.series_calls),
    )
    assert graph_probes_after > graph_probes_before, (graph_probes_before, graph_probes_after)

    # llm_usage is this request's own spend: the repeat spent nothing.
    first_usage = first["diagnostics"]["llm_usage"]
    second_usage = second["diagnostics"]["llm_usage"]
    assert first_usage is not None and first_usage["input_tokens"] > 0
    assert second_usage == {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


def test_sse_transport_shares_the_same_memo(wired):
    """A buffered call and a later SSE call share the memo while observations stream live."""
    client, replay_client, _graph, _materialise_graph, _sdmx, _capture = wired

    _body(_resolve(client, sse=False))
    calls_after_first = list(replay_client.models.calls)

    response = _resolve(client, sse=True)
    assert response.status_code == 200, response.text
    frames = parse_sse(response.text)
    assert replay_client.models.calls == calls_after_first
    assert calls_after_first == ["Extraction", "QuantityOutput", "_UniversalBindOutput"]

    event_types = [event for event, _ in frames]
    response_idx = event_types.index("response")
    assert "observations" in event_types
    obs_payload = frames[event_types.index("observations")][1]["observations"]
    assert obs_payload["total_obs"] == 3
    assert response_idx < event_types.index("observations")


def test_an_injected_llm_without_a_memo_still_calls_the_model(wired, monkeypatch):
    """An injected memo-less LLM calls Gemini on each request, unlike the app's shared memo."""
    from search.engine.clients.llm import LLM

    _client, replay_client, graph, _materialise_graph, sdmx, _capture = wired

    def _unmemoised_app():
        # A private FakeGeneration, not the None-injection default: `wired`'s
        # own app is still open (its lifespan holds the real process-wide
        # provider, registered under the same "search.*" cache names this
        # second app would also register), so leaving this one None would
        # collide registering an already-taken name onto that same shared
        # singleton.
        materialise_graph2, materialise_sdmx2 = materialise_double_from_obs(_IND_OBS)
        return create_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=LLM(),
            sdmx=sdmx,
            materialise_graph=materialise_graph2,
            materialise_sdmx=materialise_sdmx2,
            generation=FakeGeneration(),
        )

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        with TestClient(_unmemoised_app(), raise_server_exceptions=False) as plain_client:
            # The lifespan skips warm() for an injected LLM, which is the only
            # call that resolves _get_client at startup; every later call
            # resolves it through the patched module attribute.
            monkeypatch.setattr("search.engine.clients.llm._get_client", lambda: replay_client)
            _body(_resolve(plain_client, sse=False))
            _body(_resolve(plain_client, sse=False))

    assert replay_client.models.calls == [
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
        "Extraction",
        "QuantityOutput",
        "_UniversalBindOutput",
    ]


def test_a_live_answer_without_usage_metadata_is_not_logged_as_a_memo_hit(wired) -> None:
    """The envelope drops missing usage, so the memo-hit log distinguishes a live call from a hit."""
    client, replay_client, _graph, _materialise_graph, _sdmx, capture = wired

    replay_client.models.generate_content = _no_usage_generate(replay_client)

    _body(_resolve(client, sse=False))
    assert replay_client.models.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput"]
    assert "LLM memo hit" not in capture.messages

    _body(_resolve(client, sse=False))
    assert replay_client.models.calls == ["Extraction", "QuantityOutput", "_UniversalBindOutput"]
    assert "LLM memo hit: schema=QuantityOutput" in capture.messages


def _no_usage_generate(replay_client):
    """A generate_content that answers as the replaying one does, with no
    usage_metadata on the response."""
    original = replay_client.models.generate_content

    async def generate_content(*, model, contents, config):
        response = await original(model=model, contents=contents, config=config)
        response.usage_metadata = None
        return response

    return generate_content
