"""Tests the mounted `/api/search/options` route, including responses, refusals, deadlines, and wire shape."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._options_fixtures import (
    _KENYA_DCID,
    _options_client,
    _options_request,
    _pin_arm_options_request,
    _place_graph,
)
from tests.engine._search_app import search_app
from tests.fixtures import FakeLLM, make_resolver

# ---------------------------------------------------------------------------
# The mounted route: POST /api/search/options.
# ---------------------------------------------------------------------------


class TestOptionsRoute:
    def test_successful_round_trip(self):
        """A resolving identity returns the response envelope with Kenya's option."""
        body = _options_request(entities={"observationAbout": [_KENYA_DCID]}).model_dump(
            by_alias=True, mode="json"
        )
        resp = _options_client().post("/api/search/options", json=body)
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["for"] == {"role": "observationAbout"}  # scope unset -> omitted, not null
        assert "diagnostics" in payload
        dcids = {i["value"]["ref"]["dcid"] for i in payload["options"]["items"]}
        assert dcids == {_KENYA_DCID}

    def test_refusal_surfaces_as_problem_body(self):
        """A pinned-ask refusal returns a shared RFC 9457 problem body."""
        body = _pin_arm_options_request().model_dump(by_alias=True, mode="json")
        resp = _options_client().post("/api/search/options", json=body)
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/problem+json")
        assert resp.json()["code"] == "request_invalid"

    def test_elapsed_request_deadline_refuses_504(self, search_settings):
        """Both off-loop route steps share one request deadline.

        The minimum positive timeout expires before either step, without a slow fixture."""
        search_settings(request_timeout_s=1e-12)
        body = _options_request(entities={"observationAbout": [_KENYA_DCID]}).model_dump(
            by_alias=True, mode="json"
        )
        resp = _options_client().post("/api/search/options", json=body)
        assert resp.status_code == 504
        assert resp.headers["content-type"].startswith("application/problem+json")
        payload = resp.json()
        assert payload["code"] == "timeout"
        assert payload["retryable"] is True

    def test_option_target_invalid_reaches_the_wire_as_422(self):
        """A role the resolved population does not declare (see
        options.resolve's own option_target_invalid check) reaches the wire
        as the contract's own 422, not a generic 400."""
        body = _options_request(
            entities={"observationAbout": [_KENYA_DCID]}, role="destination"
        ).model_dump(by_alias=True, mode="json")
        resp = _options_client().post("/api/search/options", json=body)
        assert resp.status_code == 422
        assert resp.json()["code"] == "option_target_invalid"

    def test_materialise_graph_uninitialised_refuses_503(self):
        """Every options request needs a materialise_graph: population
        derivation is load-bearing, so app.py's own route-entry None guard
        refuses before resolve_options_target ever runs."""
        body = _options_request(entities={"observationAbout": [_KENYA_DCID]}).model_dump(
            by_alias=True, mode="json"
        )
        graph = _place_graph()
        client = TestClient(
            search_app(
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}),
            ),
            raise_server_exceptions=False,
        )
        resp = client.post("/api/search/options", json=body)
        assert resp.status_code == 503
        assert resp.json()["code"] == "service_unavailable"


def _nulls(payload, path: str = "") -> list[str]:
    """Every JSON path in `payload` whose value is null."""
    if payload is None:
        return [path or "<root>"]
    if isinstance(payload, dict):
        return [p for k, v in payload.items() for p in _nulls(v, f"{path}/{k}")]
    if isinstance(payload, list):
        return [p for i, v in enumerate(payload) for p in _nulls(v, f"{path}/{i}")]
    return []


class TestOptionsRouteWireShape:
    """Checks JSON wire shape. Absent fields mean unrequested.
    SSE uses null for unconfirmed coverage."""

    def test_no_absent_member_is_emitted_as_null(self):
        """A blanket check rather than a per-field list, so a new optional
        member added to any model in this envelope is covered by default
        instead of needing its own case here."""
        body = _options_request(entities={"observationAbout": [_KENYA_DCID]}).model_dump(
            by_alias=True, mode="json"
        )
        resp = _options_client().post("/api/search/options", json=body)
        assert resp.status_code == 200
        assert _nulls(resp.json()) == []

    def test_unprobed_coverage_is_omitted_not_null(self):
        """The default request never probes coverage (include_coverage
        defaults false), so emitting `coverage: null` per item would claim a
        probe that never ran."""
        body = _options_request(entities={"observationAbout": [_KENYA_DCID]}).model_dump(
            by_alias=True, mode="json"
        )
        resp = _options_client().post("/api/search/options", json=body)
        items = resp.json()["options"]["items"]
        assert items, "fixture must return at least one item for this to assert anything"
        assert all("coverage" not in item for item in items)

    def test_requested_coverage_is_present_on_the_wire(self):
        """The other half of the contract: omission means "not requested", so
        asking for coverage must actually put it in the body -- otherwise the
        test above would pass on a response that dropped it wholesale."""
        body = _options_request(
            entities={"observationAbout": [_KENYA_DCID]}, include_coverage=True
        ).model_dump(by_alias=True, mode="json")
        resp = _options_client().post("/api/search/options", json=body)
        assert resp.status_code == 200
        items = resp.json()["options"]["items"]
        assert items
        assert all("coverage" in item for item in items)
        assert _nulls(resp.json()) == []

    def test_for_is_serialised_under_its_alias(self):
        """`for_` is a reserved-word field name aliased to the contract's
        `for`; the field name must never reach the wire."""
        body = _options_request(entities={"observationAbout": [_KENYA_DCID]}).model_dump(
            by_alias=True, mode="json"
        )
        resp = _options_client().post("/api/search/options", json=body)
        assert resp.status_code == 200
        payload = resp.json()
        assert "for" in payload
        assert "for_" not in payload

    def test_requested_scope_is_echoed_on_the_wire(self):
        """Omission must be the only reason a member is absent: an explicitly
        requested scope round-trips into the echoed `for`."""
        body = _options_request(
            entities={"observationAbout": [_KENYA_DCID]}, scope="peers"
        ).model_dump(by_alias=True, mode="json")
        resp = _options_client().post("/api/search/options", json=body)
        assert resp.status_code == 200
        assert resp.json()["for"] == {"role": "observationAbout", "scope": "peers"}
