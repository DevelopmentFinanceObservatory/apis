"""Contract tests for the omission reason vocabulary, OpenAPI fields, and SSE
wire shape, so `website` can build `types.ts` against a fixture. Payloads use
the real constructors, app entry points, and encoder.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import types
from datetime import date
from typing import get_args
from unittest.mock import patch

from search.engine import enrichment
from search.engine.app import create_app
from search.engine.codes import WARNING_CODES
from search.engine.core import resolve_events
from search.engine.events import ObservationsEvent, serialize_sse
from search.engine.region.codes import (
    OBSERVATIONS_NO_POINTS,
    OBSERVATIONS_TIMED_OUT,
)
from search.models import RawTextInput, ResolveOptions, ResolveRequest
from search.models.observations import omitted_observations
from search.models.vocab import ObservationsOmissionReason
from search.observations import OBSERVATIONS_PREVIEW_TOO_LARGE
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import PINNED_DATE, parse_sse
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

# The closed vocabulary as declared in search.models.vocab -- hardcoded (not
# re-derived from get_args) so a reason added, removed, or renamed there
# without a matching update here fails this test rather than passing by
# construction.
_EXPECTED_REASONS = frozenset(
    {
        "observations_timed_out",
        "observations_no_matching_source",
        "observations_windowed_scope_unsupported",
        "observations_converted_scope_unsupported",
        "observations_no_dispatchable_facet",
        "observations_bulk_fetch_failed",
        "observations_probe_failed",
        "observations_preview_too_large",
        "observations_nothing_to_probe",
        "observations_clients_unavailable",
        "observations_no_points",
    }
)


def _schema() -> dict:
    """The composed OpenAPI schema for the search engine surface, built with
    no live credential -- app.openapi() needs no lifespan/materialise client,
    same as test_openapi_response_models.py's own `schema` fixture."""
    graph = FakeGraph()
    app = create_app(graph=graph, resolver=make_resolver(graph), llm=FakeLLM())
    return app.openapi()


# ---------------------------------------------------------------------------
# 1. The Literal reason vocabulary and the ObservationsOmission field set,
#    pinned in /openapi.json -- what `website`'s types.ts diffs against.
# ---------------------------------------------------------------------------


class TestObservationsOmissionOpenAPISchema:
    def test_reason_vocabulary_is_pinned(self):
        # Pydantic inlines this `Literal` as the `reason` property's enum.
        schema = _schema()
        omission = schema["components"]["schemas"]["ObservationsOmission"]
        reason_schema = omission["properties"]["reason"]
        assert reason_schema["type"] == "string"
        assert set(reason_schema["enum"]) == _EXPECTED_REASONS
        assert len(reason_schema["enum"]) == len(_EXPECTED_REASONS), (
            "duplicate reason literal in the schema's enum"
        )

    def test_reason_vocabulary_matches_the_vocab_module(self):
        # Keep the hardcoded OpenAPI pin in sync with the vocabulary alias.
        assert set(get_args(ObservationsOmissionReason)) == _EXPECTED_REASONS

    def test_every_reason_is_registered_as_a_warning_code(self):
        # Models and engine register reasons separately, so the test checks
        # each omission reason against the warning registry.
        assert set(get_args(ObservationsOmissionReason)) <= WARNING_CODES

    def test_field_set_is_pinned(self):
        schema = _schema()
        omission = schema["components"]["schemas"]["ObservationsOmission"]
        assert set(omission["properties"]) == {
            "reason",
            "message",
            "pairs",
            "cap",
            "dispatched",
            "empty",
            "hard_failed",
            "facet_rejected",
        }
        # Reject fields absent from the pinned wire shape.
        assert omission["additionalProperties"] is False
        assert omission["required"] == ["reason", "message"]
        for optional_field in (
            "pairs",
            "cap",
            "dispatched",
            "empty",
            "hard_failed",
            "facet_rejected",
        ):
            branches = omission["properties"][optional_field]["anyOf"]
            assert any(branch.get("type") == "integer" for branch in branches), optional_field
            assert any(branch.get("type") == "null" for branch in branches), optional_field

    def test_spec_observations_omitted_is_an_optional_ref(self):
        schema = _schema()
        spec_observations = schema["components"]["schemas"]["SpecObservations"]
        omitted = spec_observations["properties"]["omitted"]
        refs = {branch["$ref"] for branch in omitted["anyOf"] if "$ref" in branch}
        assert refs == {"#/components/schemas/ObservationsOmission"}
        assert any(branch.get("type") == "null" for branch in omitted["anyOf"])
        assert "omitted" not in spec_observations.get("required", [])


# ---------------------------------------------------------------------------
# 2. The structural invariant, for every reason in the closed vocabulary:
#    omitted set => series=[], total_obs=0, truncated=False.
# ---------------------------------------------------------------------------


def test_every_reason_forces_series_empty_total_zero_truncated_false():
    for reason in get_args(ObservationsOmissionReason):
        observations = omitted_observations(reason, f"contract fixture message for {reason}")
        assert observations.series == [], reason
        assert observations.total_obs == 0, reason
        assert observations.truncated is False, reason
        assert observations.omitted is not None, reason
        assert observations.omitted.reason == reason, reason


# ---------------------------------------------------------------------------
# 3. Field-by-field wire capture: the real ObservationsEvent, through the
#    real serialize_sse encoder the app uses for every SSE frame.
# ---------------------------------------------------------------------------


def _captured_frame(observations, *, spec_id: str = "s-contract") -> dict:
    event = ObservationsEvent(spec_id=spec_id, observations=observations)
    frame = serialize_sse(event)
    assert frame.startswith("event: observations\ndata: ")
    assert frame.endswith("\n\n")
    data_line = frame.split("\n", 1)[1][len("data: ") :].rstrip("\n")
    return json.loads(data_line)


def test_preview_too_large_wire_shape_field_by_field():
    # 126 members against a 120 cap: the same numbers the
    # education/Kenya/2024 live-confirmation scenario uses.
    observations = omitted_observations(
        OBSERVATIONS_PREVIEW_TOO_LARGE,
        "This selection has 126 series, above the 120 the preview fetches. Narrow the query.",
        pairs=126,
        cap=120,
    )
    payload = _captured_frame(observations)
    assert payload == {
        "kind": "observations",
        "spec_id": "s-contract",
        "observations": {
            "series": [],
            "total_obs": 0,
            "truncated": False,
            "omitted": {
                "reason": "observations_preview_too_large",
                "message": (
                    "This selection has 126 series, above the 120 the preview fetches. "
                    "Narrow the query."
                ),
                "pairs": 126,
                "cap": 120,
            },
        },
    }
    # Unset counts are absent from the serialized payload, not null.
    assert "dispatched" not in payload["observations"]["omitted"]
    assert "empty" not in payload["observations"]["omitted"]
    assert "hard_failed" not in payload["observations"]["omitted"]
    assert "facet_rejected" not in payload["observations"]["omitted"]


def test_no_points_wire_shape_field_by_field():
    observations = omitted_observations(
        OBSERVATIONS_NO_POINTS,
        "153 observations are recorded but none returned in the preview.",
        dispatched=5,
        empty=3,
        hard_failed=0,
        facet_rejected=2,
    )
    payload = _captured_frame(observations)
    assert payload == {
        "kind": "observations",
        "spec_id": "s-contract",
        "observations": {
            "series": [],
            "total_obs": 0,
            "truncated": False,
            "omitted": {
                "reason": "observations_no_points",
                "message": "153 observations are recorded but none returned in the preview.",
                "dispatched": 5,
                "empty": 3,
                "hard_failed": 0,
                "facet_rejected": 2,
            },
        },
    }
    assert "pairs" not in payload["observations"]["omitted"]
    assert "cap" not in payload["observations"]["omitted"]


def test_omitted_payload_round_trips_through_observationsevent():
    for reason, kwargs in (
        (OBSERVATIONS_PREVIEW_TOO_LARGE, {"pairs": 126, "cap": 120}),
        (
            OBSERVATIONS_NO_POINTS,
            {"dispatched": 5, "empty": 3, "hard_failed": 0, "facet_rejected": 2},
        ),
    ):
        observations = omitted_observations(reason, "msg", **kwargs)
        event = ObservationsEvent(spec_id="s-roundtrip", observations=observations)
        frame = serialize_sse(event)
        data_line = frame.split("\n", 1)[1][len("data: ") :].rstrip("\n")
        rebuilt = ObservationsEvent.model_validate_json(data_line)
        assert rebuilt.observations.omitted == observations.omitted
        assert rebuilt.observations.series == []
        assert rebuilt.observations.total_obs == 0
        assert rebuilt.observations.truncated is False


# ---------------------------------------------------------------------------
# 4. The always-emitted invariant, at the contract level: driven through the
#    real pipeline (resolve_events), not a hand-built RegionResult. A
#    definite spec that requested observations must carry exactly one
#    ObservationsEvent even when the fetch itself never returns before the
#    request deadline.
# ---------------------------------------------------------------------------


def test_deadline_synthesized_event_is_emitted_through_the_real_pipeline(
    monkeypatch, search_settings
):
    """Through the real pipeline, a timed-out observation fetch emits one
    `ObservationsEvent` before `done`.

    The resolve stages run on the real clock under the default deadline, so the
    leg always commits its definite Spec. The trailing-enrichment clock reads one
    request timeout ahead, so the observation fetch, held until the stream ends,
    is past the request deadline when the stream first checks it.
    """
    released = threading.Event()

    def _held_observations_events(*args, **kwargs):
        released.wait(timeout=30)
        raise AssertionError("should have been cancelled by the request deadline")

    # Hold the trailing read alone. The answer phase's own read of the same preview, which
    # states the immediate response's facets, runs as usual.
    monkeypatch.setattr(
        "search.engine.region.observations.dispatch._observations_events",
        _held_observations_events,
    )
    timeout_s = search_settings().request_timeout_s
    monkeypatch.setattr(
        enrichment,
        "time",
        types.SimpleNamespace(monotonic=lambda: time.monotonic() + timeout_s),
    )

    async def _body():
        try:
            return [
                ev
                async for ev in resolve_events(
                    ResolveRequest(
                        input=RawTextInput(query="total population India"),
                        response_options=ResolveOptions(include_observations=True),
                    ),
                    graph=FakeGraph(),
                    llm=FakeLLM(),
                    sdmx=FakeSdmxClient(),
                    stream_slot_options=True,
                )
            ]
        finally:
            # Free the worker thread before `asyncio.run` waits for its executor.
            released.set()

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        events = asyncio.run(_body())

    kinds = [e.kind for e in events]
    assert kinds.count("response") == 1
    assert kinds.count("done") == 1
    obs_events = [e for e in events if e.kind == "observations"]
    assert len(obs_events) == 1, (
        f"expected exactly one ObservationsEvent for the definite, "
        f"observations-requesting spec, even on deadline; got {len(obs_events)} "
        f"among event kinds {kinds}"
    )
    (obs_event,) = obs_events
    assert obs_event.observations.omitted is not None
    assert obs_event.observations.omitted.reason == OBSERVATIONS_TIMED_OUT
    assert obs_event.observations.series == []
    assert obs_event.observations.total_obs == 0
    assert obs_event.observations.truncated is False

    response_idx = kinds.index("response")
    done_idx = kinds.index("done")
    obs_idx = kinds.index("observations")
    assert response_idx < obs_idx < done_idx

    done_event = events[done_idx]
    assert any(w.code == OBSERVATIONS_TIMED_OUT for w in done_event.warnings)

    # The same synthesized event, captured through the real wire encoder --
    # ties this pipeline-level proof back to the field-by-field pin above.
    frame = serialize_sse(obs_event)
    frames = parse_sse(frame)
    ((sse_kind, sse_payload),) = frames
    assert sse_kind == "observations"
    assert sse_payload["observations"]["omitted"]["reason"] == "observations_timed_out"
    assert sse_payload["observations"]["series"] == []
