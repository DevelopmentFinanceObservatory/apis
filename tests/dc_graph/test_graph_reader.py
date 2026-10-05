"""Request-shape and response-parsing tests for `GraphClient`."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

import httpx
import pytest

import dc_graph.client as graph_module
from dc_graph import (
    UNRESOLVED_NODE,
    Facet,
    GraphClient,
    GraphDeadlineExceededError,
    GraphNotAdvancingError,
    GraphReadError,
    SubjectPage,
)
from dc_graph.client import _arc_spec

from .conftest import read_fixture_text


def _reader(
    transport: httpx.BaseTransport, *, timeout: float = 30.0, max_attempts: int = 3
) -> GraphClient:
    return GraphClient(
        base_url="https://graph.example.org/core/api",
        timeout=timeout,
        transport=transport,
        max_attempts=max_attempts,
    )


def _replay(body: str, *, status_code: int = 200, content_type: str = "application/json"):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, content=body, headers={"content-type": content_type})

    return httpx.MockTransport(handler), requests


class _RecordingTransport(httpx.BaseTransport):
    """Captures the last request it served, so a test can inspect the
    per-request httpx timeout `post` actually applied.
    """

    def __init__(self, body: str = "{}") -> None:
        self.body = body
        self.request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(200, content=self.body, headers={"content-type": "application/json"})


def _sequenced_transport(
    responses: list[httpx.Response | Exception],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A transport that serves `responses` in order, one per request,
    raising an exception in place of returning it when an element is one.
    """
    requests: list[httpx.Request] = []
    remaining = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = next(remaining)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.MockTransport(handler), requests


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # `post`'s retry backoff sleeps for real seconds by default. Patch it
    # to a no-op so this suite's runtime doesn't grow whenever a test's
    # mocked transport happens to return a retryable failure. A test that
    # asserts on the backoff itself installs its own recording replacement,
    # which simply overrides this one for that test's duration.
    monkeypatch.setattr(graph_module.time, "sleep", lambda seconds: None)


def test_node_label_request_shape():
    body = read_fixture_text("node_label_found.json")
    transport, requests = _replay(body)
    reader = _reader(transport)

    label = reader.node_label("country/KEN")

    assert label == "Kenya"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/core/api/v2/node"
    payload = json.loads(request.content)
    assert payload == {"nodes": ["country/KEN"], "property": "->name"}


def test_node_label_absent_node_returns_none_not_raises():
    body = read_fixture_text("node_label_absent.json")
    transport, _ = _replay(body)
    reader = _reader(transport)

    assert reader.node_label("country/ZZZ") is None


def test_observation_facets_batch_request_shape_and_parsing():
    # `/v2/observation` uses bare StatVar dcids; this request-shape test pins the payload.
    body = read_fixture_text("observation_facets_found.json")
    transport, requests = _replay(body)
    reader = _reader(transport)

    facets = reader.variable_facets_batch(
        stat_var="MalariaGrantDisbursements", entities=["country/KEN"]
    )

    assert facets == {
        "country/KEN": [
            Facet(
                facet_id="facet1",
                provenance_id="OECD_DAC2A",
                measurement_method="GrantEquivalent",
                unit="USD",
                obs_count=402,
                earliest_date="1991",
                latest_date="2024",
            )
        ]
    }
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/core/api/v2/observation"
    payload = json.loads(request.content)
    assert payload == {
        "select": ["variable", "entity", "facet"],
        "variable": {"dcids": ["MalariaGrantDisbursements"]},
        "entity": {"dcids": ["country/KEN"]},
    }


def test_observation_facets_batch_captures_scaling_factor():
    body = read_fixture_text("observation_facets_scaling_factor.json")
    transport, _ = _replay(body)
    reader = _reader(transport)

    facets = reader.variable_facets_batch(
        stat_var="dcid:MalariaGrantDisbursements", entities=["country/KEN"]
    )

    assert facets["country/KEN"][0].scaling_factor == "100"


def test_observation_facets_batch_absent_scaling_factor_is_none_not_empty_string():
    body = read_fixture_text("observation_facets_found.json")
    transport, _ = _replay(body)
    reader = _reader(transport)

    facets = reader.variable_facets_batch(
        stat_var="MalariaGrantDisbursements", entities=["country/KEN"]
    )

    assert facets["country/KEN"][0].scaling_factor is None


def test_observation_facets_batch_captures_observation_period():
    # (#439) `observationPeriod` is in the top-level `facets` map, beside the unit.
    body = read_fixture_text("observation_facets_observation_period.json")
    transport, _ = _replay(body)
    reader = _reader(transport)

    facets = reader.variable_facets_batch(
        stat_var="MalariaGrantDisbursements", entities=["country/KEN"]
    )

    assert facets["country/KEN"][0].observation_period == "P1Y"


def test_observation_facets_batch_absent_observation_period_is_none():
    body = read_fixture_text("observation_facets_found.json")
    transport, _ = _replay(body)
    reader = _reader(transport)

    facets = reader.variable_facets_batch(
        stat_var="MalariaGrantDisbursements", entities=["country/KEN"]
    )

    assert facets["country/KEN"][0].observation_period is None


def test_observation_facets_batch_multi_entity_one_post():
    body = read_fixture_text("observation_facets_found.json")
    transport, requests = _replay(body)
    reader = _reader(transport)

    facets = reader.variable_facets_batch(
        stat_var="MalariaGrantDisbursements", entities=["country/KEN", "country/TGO"]
    )

    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["entity"] == {"dcids": ["country/KEN", "country/TGO"]}
    assert facets["country/KEN"] != []
    assert facets["country/TGO"] == []  # absent from the fixture's byEntity map


def test_observation_facets_batch_empty_entities_returns_empty_dict_no_call():
    transport, requests = _replay("{}")
    reader = _reader(transport)

    assert reader.variable_facets_batch(stat_var="dcid:Whatever", entities=[]) == {}
    assert requests == []


def test_observation_facets_batch_no_observations_returns_empty_list():
    body = read_fixture_text("observation_facets_empty.json")
    transport, _ = _replay(body)
    reader = _reader(transport)

    facets = reader.variable_facets_batch(stat_var="dcid:Whatever", entities=["country/KEN"])

    assert facets == {"country/KEN": []}


def test_observation_facets_batch_merges_blocks_sharing_a_facet_id():
    """A facet list from `variable_facets_batch` carries each facetId at
    most once: two blocks tagged "A" merge into one Facet (obs_count
    summed, earliest/latest widened), "B" stays its own entry, and the
    id-less block stays separate rather than folding into either."""
    body = read_fixture_text("observation_facets_duplicate_facet_id.json")
    transport, _ = _replay(body)
    reader = _reader(transport)

    facets = reader.variable_facets_batch(
        stat_var="Amount_SyntheticFlow_Test", entities=["country/ZZZ"]
    )["country/ZZZ"]

    facet_ids = [f.facet_id for f in facets]
    assert len(facet_ids) == len(set(facet_ids))
    assert facet_ids == ["A", "B", None]
    merged_a = facets[0]
    assert merged_a.obs_count == 3
    assert merged_a.earliest_date == "2019"
    assert merged_a.latest_date == "2021"
    assert facets[1].obs_count == 1
    assert facets[2].obs_count == 1


def test_observation_facets_batch_many_request_shape_and_parsing():
    """One POST carries all StatVars and entities; the result is split by `(stat_var, entity)`."""
    body = json.dumps(
        {
            "byVariable": {
                "MalariaGrantDisbursements": {
                    "byEntity": {
                        "country/KEN": {
                            "orderedFacets": [
                                {
                                    "facetId": "facet1",
                                    "obsCount": 402,
                                    "earliestDate": "1991",
                                    "latestDate": "2024",
                                }
                            ]
                        }
                    }
                },
                "GDP": {
                    "byEntity": {
                        "country/TGO": {
                            "orderedFacets": [
                                {
                                    "facetId": "facet2",
                                    "obsCount": 10,
                                    "earliestDate": "2000",
                                    "latestDate": "2020",
                                }
                            ]
                        }
                    }
                },
            },
            "facets": {
                "facet1": {
                    "provenanceId": "OECD_DAC2A",
                    "measurementMethod": "GrantEquivalent",
                    "unit": "USD",
                },
                "facet2": {
                    "provenanceId": "WorldBank",
                    "measurementMethod": None,
                    "unit": "USDollar",
                },
            },
        }
    )
    transport, requests = _replay(body)
    reader = _reader(transport)

    result = reader.observation_facets_batch_many(
        stat_vars=["MalariaGrantDisbursements", "GDP"],
        entities=["country/KEN", "country/TGO"],
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/core/api/v2/observation"
    payload = json.loads(request.content)
    assert payload == {
        "select": ["variable", "entity", "facet"],
        "variable": {"dcids": ["MalariaGrantDisbursements", "GDP"]},
        "entity": {"dcids": ["country/KEN", "country/TGO"]},
    }

    assert result["MalariaGrantDisbursements"]["country/KEN"] == [
        Facet(
            facet_id="facet1",
            provenance_id="OECD_DAC2A",
            measurement_method="GrantEquivalent",
            unit="USD",
            obs_count=402,
            earliest_date="1991",
            latest_date="2024",
        )
    ]
    # Unpopulated StatVar/entity pairs remain present with empty lists.
    assert result["MalariaGrantDisbursements"]["country/TGO"] == []
    assert result["GDP"]["country/KEN"] == []
    assert result["GDP"]["country/TGO"][0].facet_id == "facet2"


def test_observation_facets_batch_many_empty_stat_vars_returns_empty_dict_no_call():
    transport, requests = _replay("{}")
    reader = _reader(transport)

    assert reader.observation_facets_batch_many(stat_vars=[], entities=["country/KEN"]) == {}
    assert requests == []


def test_observation_facets_batch_many_empty_entities_returns_empty_dict_no_call():
    transport, requests = _replay("{}")
    reader = _reader(transport)

    result = reader.observation_facets_batch_many(stat_vars=["dcid:Whatever"], entities=[])

    assert result == {}
    assert requests == []


def test_deadline_uses_the_lower_of_deadline_and_configured_timeout():
    # The shorter deadline bounds the request timeout.
    short_deadline_transport = _RecordingTransport()
    short_deadline_reader = _reader(short_deadline_transport, timeout=90)
    short_deadline_reader.variable_facets_batch(
        stat_var="dcid:Whatever", entities=["country/KEN"], deadline=time.monotonic() + 5
    )
    assert short_deadline_transport.request is not None
    assert 0 < short_deadline_transport.request.extensions["timeout"]["read"] <= 5

    # The configured timeout remains the bound when shorter.
    long_deadline_transport = _RecordingTransport()
    long_deadline_reader = _reader(long_deadline_transport, timeout=2)
    long_deadline_reader.variable_facets_batch(
        stat_var="dcid:Whatever", entities=["country/KEN"], deadline=time.monotonic() + 30
    )
    assert long_deadline_transport.request is not None
    assert long_deadline_transport.request.extensions["timeout"]["read"] == 2


def test_expired_deadline_raises_before_any_request():
    transport = _RecordingTransport()
    reader = _reader(transport)

    with pytest.raises(GraphDeadlineExceededError, match="absolute caller deadline"):
        reader.variable_facets_batch(
            stat_var="dcid:Whatever", entities=["country/KEN"], deadline=time.monotonic() - 1
        )

    assert transport.request is None


def test_timeout_exception_under_a_deadline_raises_graph_deadline_exceeded_error():
    # The 5-second deadline, not the 30-second client timeout, bounds this attempt.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated slow response", request=request)

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphDeadlineExceededError):
        reader.variable_facets_batch(
            stat_var="dcid:Whatever", entities=["country/KEN"], deadline=time.monotonic() + 5
        )


def test_post_retries_a_timeout_with_budget_left_then_succeeds(monkeypatch):
    # The 5-second timeout retries while 30 seconds remain.
    ok_body = read_fixture_text("observation_facets_found.json")
    transport, requests = _sequenced_transport(
        [
            httpx.ReadTimeout("simulated slow response"),
            httpx.Response(200, content=ok_body, headers={"content-type": "application/json"}),
        ]
    )
    reader = _reader(transport, timeout=5.0)
    sleeps: list[float] = []
    monkeypatch.setattr(graph_module.time, "sleep", sleeps.append)

    facets = reader.variable_facets_batch(
        stat_var="MalariaGrantDisbursements",
        entities=["country/KEN"],
        deadline=time.monotonic() + 30.0,
    )

    assert len(requests) == 2
    assert sleeps == [graph_module._RETRY_BACKOFF_SECONDS[0]]
    assert facets["country/KEN"] != []


def test_post_timeout_with_budget_spent_raises_graph_deadline_exceeded(monkeypatch):
    # The 5-second deadline bounds the attempt and prevents a retry after expiry.
    transport, requests = _sequenced_transport([httpx.ReadTimeout("simulated slow response")])
    reader = _reader(transport)
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: 0.0)

    with pytest.raises(GraphDeadlineExceededError, match="absolute caller deadline"):
        reader.variable_facets_batch(
            stat_var="dcid:Whatever", entities=["country/KEN"], deadline=5.0
        )

    assert len(requests) == 1


def test_post_repeated_timeouts_with_budget_left_exhaust_retries(monkeypatch):
    # All three transport timeouts occur with deadline budget left and exhaust normal retries.
    transport, requests = _sequenced_transport(
        [
            httpx.ReadTimeout("simulated slow response"),
            httpx.ReadTimeout("simulated slow response"),
            httpx.ReadTimeout("simulated slow response"),
        ]
    )
    reader = _reader(transport, timeout=5.0)  # default max_attempts=3
    sleeps: list[float] = []
    monkeypatch.setattr(graph_module.time, "sleep", sleeps.append)

    with pytest.raises(GraphReadError) as excinfo:
        reader.variable_facets_batch(
            stat_var="dcid:Whatever",
            entities=["country/KEN"],
            deadline=time.monotonic() + 30.0,
        )

    assert not isinstance(excinfo.value, GraphDeadlineExceededError)
    assert len(requests) == 3
    assert sleeps == list(graph_module._RETRY_BACKOFF_SECONDS)


def test_timeout_exception_with_no_deadline_raises_plain_graph_read_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated slow response", request=request)

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphReadError) as excinfo:
        reader.variable_facets_batch(stat_var="dcid:Whatever", entities=["country/KEN"])
    assert not isinstance(excinfo.value, GraphDeadlineExceededError)


def test_non_2xx_response_raises_graph_read_error():
    transport, _ = _replay("internal error", status_code=500, content_type="text/plain")
    reader = _reader(transport)

    with pytest.raises(GraphReadError) as excinfo:
        reader.node_label("country/KEN")
    assert excinfo.value.upstream_status == 500


def test_non_json_response_raises_graph_read_error():
    transport, _ = _replay("not json", status_code=200, content_type="text/plain")
    reader = _reader(transport)

    with pytest.raises(GraphReadError):
        reader.variable_facets_batch(stat_var="dcid:Whatever", entities=["country/KEN"])


def test_transport_error_raises_graph_read_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    reader = _reader(transport)

    with pytest.raises(GraphReadError):
        reader.node_label("country/KEN")


def test_default_facet_policy_always_none():
    transport, requests = _replay("{}")
    reader = _reader(transport)

    assert reader.default_facet_policy(stat_var="dcid:Whatever", axis="unit") is None
    assert reader.default_facet_policy(stat_var="dcid:Other", axis="provenance") is None
    # Returns None without making a network call.
    assert requests == []


def test_node_arcs_many_request_shape():
    body = read_fixture_text("node_arcs_many_parse.json")
    transport, requests = _replay(body)
    reader = _reader(transport)

    reader.node_arcs_many(
        ["country/KEN", "country/ZZZ"], properties=("typeOf", "name", "description")
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/core/api/v2/node"
    payload = json.loads(request.content)
    assert payload == {
        "nodes": ["country/KEN", "country/ZZZ"],
        "property": "->[typeOf,name,description]",
    }


def test_node_arcs_many_parses_held_absent_and_missing_property():
    # The empty `country/ZZZ` result is retried alone; missing `description` maps to `()`.
    first_body = json.dumps(
        {
            "data": {
                "country/KEN": {
                    "arcs": {
                        "typeOf": {"nodes": [{"dcid": "Country"}]},
                        "name": {"nodes": [{"value": "Kenya"}]},
                    }
                },
                "country/ZZZ": {},
            }
        }
    )
    second_body = json.dumps({"data": {"country/ZZZ": {}}})
    third_body = json.dumps({"data": {"country/ZZZ": {}}})
    responses = [first_body, second_body, third_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.node_arcs_many(
        ["country/KEN", "country/ZZZ"], properties=("typeOf", "name", "description")
    )

    assert len(requests) == 3
    second_payload = json.loads(requests[1].content)
    assert second_payload["nodes"] == ["country/ZZZ"]
    third_payload = json.loads(requests[2].content)
    assert third_payload["nodes"] == ["country/ZZZ"]
    assert result == {
        "country/KEN": {"typeOf": ("Country",), "name": ("Kenya",), "description": ()},
        "country/ZZZ": None,
    }


def test_node_arcs_many_single_dcid_present_but_empty_trusted_without_retry():
    # A lone present-but-empty dcid is trusted in one POST.
    body = json.dumps({"data": {"country/ZZZ": {}}})
    transport, requests = _replay(body)
    reader = _reader(transport)

    result = reader.node_arcs_many(["country/ZZZ"], properties=("typeOf",))

    assert result == {"country/ZZZ": None}
    assert len(requests) == 1


def test_node_arcs_many_single_dcid_missing_key_still_retries():
    # A missing response key gets a singleton retry, then a probe if still empty.
    first_body = json.dumps({"data": {}})
    second_body = json.dumps({"data": {"country/ZZZ": {}}})
    third_body = json.dumps({"data": {"country/ZZZ": {}}})
    responses = [first_body, second_body, third_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.node_arcs_many(["country/ZZZ"], properties=("typeOf",))

    assert len(requests) == 3
    second_payload = json.loads(requests[1].content)
    assert second_payload["nodes"] == ["country/ZZZ"]
    third_payload = json.loads(requests[2].content)
    assert third_payload["nodes"] == ["country/ZZZ"]
    assert result == {"country/ZZZ": None}


def test_node_arcs_many_multi_dcid_false_empty_caught_by_retry():
    # The initial batch reports "country/ZZZ" present but empty. The retry,
    # scoped to just that dcid, reveals it actually carries arcs. The
    # false-empty the initial batch's composition produced is caught.
    first_body = json.dumps(
        {
            "data": {
                "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                "country/ZZZ": {},
            }
        }
    )
    second_body = json.dumps(
        {"data": {"country/ZZZ": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.node_arcs_many(["country/KEN", "country/ZZZ"], properties=("typeOf",))

    assert len(requests) == 2
    assert result == {
        "country/KEN": {"typeOf": ("Country",)},
        "country/ZZZ": {"typeOf": ("Country",)},
    }


def test_node_arcs_many_group_retry_false_empty_caught_by_individual_probe():
    # Both "country/ZZZ" and "country/YYY" come back present but empty in
    # the initial batch and in the group retry that re-batches them
    # together. A false-empty shared by both suspects in that one retry
    # batch, which the group retry alone cannot tell apart from a genuine
    # absence. Each gets its own individual, single-dcid probe, which
    # reveals both actually carry arcs.
    first_body = json.dumps(
        {
            "data": {
                "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                "country/ZZZ": {},
                "country/YYY": {},
            }
        }
    )
    group_retry_body = json.dumps({"data": {"country/ZZZ": {}, "country/YYY": {}}})
    zzz_probe_body = json.dumps(
        {"data": {"country/ZZZ": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    yyy_probe_body = json.dumps(
        {"data": {"country/YYY": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    responses = [first_body, group_retry_body, zzz_probe_body, yyy_probe_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.node_arcs_many(
        ["country/KEN", "country/ZZZ", "country/YYY"], properties=("typeOf",)
    )

    assert len(requests) == 4
    group_retry_payload = json.loads(requests[1].content)
    assert group_retry_payload["nodes"] == ["country/ZZZ", "country/YYY"]
    assert json.loads(requests[2].content)["nodes"] == ["country/ZZZ"]
    assert json.loads(requests[3].content)["nodes"] == ["country/YYY"]
    assert result == {
        "country/KEN": {"typeOf": ("Country",)},
        "country/ZZZ": {"typeOf": ("Country",)},
        "country/YYY": {"typeOf": ("Country",)},
    }


def test_node_arcs_many_individual_probe_missing_key_raises():
    # The individual probe omits "country/ZZZ" from "data" entirely, rather
    # than reporting it present-but-empty. This is a protocol violation, the
    # same as a group-retry-missing dcid.
    first_body = json.dumps(
        {
            "data": {
                "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                "country/ZZZ": {},
            }
        }
    )
    group_retry_body = json.dumps({"data": {"country/ZZZ": {}}})
    probe_body = json.dumps({"data": {}})
    responses = [first_body, group_retry_body, probe_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphReadError):
        reader.node_arcs_many(["country/KEN", "country/ZZZ"], properties=("typeOf",))
    assert len(requests) == 3


def test_node_arcs_many_individual_probe_budget_stops_probing_beyond_the_cap(caplog):
    # Three suspects survive the group retry still empty. With the budget
    # capped at one, only the first gets an individual probe. The other
    # two are left unprobed, and their group-retry empty result stands,
    # resolving to `None` the same way a confirmed-absent dcid would.
    first_body = json.dumps({"data": {"country/AAA": {}, "country/BBB": {}, "country/CCC": {}}})
    group_retry_body = json.dumps(
        {"data": {"country/AAA": {}, "country/BBB": {}, "country/CCC": {}}}
    )
    probe_body = json.dumps({"data": {"country/AAA": {}}})
    responses = [first_body, group_retry_body, probe_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    with caplog.at_level("INFO", logger="dc_graph.client"):
        result = reader.node_arcs_many(
            ["country/AAA", "country/BBB", "country/CCC"],
            properties=("typeOf",),
            individual_probe_budget=1,
        )

    assert len(requests) == 3
    assert result == {"country/AAA": None, "country/BBB": None, "country/CCC": None}
    assert any("2 dcid" in record.getMessage() for record in caplog.records)


def test_node_arcs_many_default_budget_probes_every_suspect_individually():
    # With no `individual_probe_budget` passed, the default probes every
    # suspect still empty after the group retry individually, however
    # many there are.
    first_body = json.dumps({"data": {"country/AAA": {}, "country/BBB": {}}})
    group_retry_body = json.dumps({"data": {"country/AAA": {}, "country/BBB": {}}})
    aaa_probe_body = json.dumps(
        {"data": {"country/AAA": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    bbb_probe_body = json.dumps(
        {"data": {"country/BBB": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    responses = [first_body, group_retry_body, aaa_probe_body, bbb_probe_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.node_arcs_many(["country/AAA", "country/BBB"], properties=("typeOf",))

    assert len(requests) == 4
    assert result == {
        "country/AAA": {"typeOf": ("Country",)},
        "country/BBB": {"typeOf": ("Country",)},
    }


def test_node_arcs_many_expired_deadline_raises_before_any_request():
    transport = _RecordingTransport()
    reader = _reader(transport)

    with pytest.raises(GraphDeadlineExceededError, match="absolute caller deadline"):
        reader.node_arcs_many(
            ["country/KEN"], properties=("typeOf",), deadline=time.monotonic() - 1
        )

    assert transport.request is None


def test_node_arcs_many_deadline_none_behaves_exactly_as_before(monkeypatch: pytest.MonkeyPatch):
    # No `deadline` given issues the request through the client's own
    # configured timeout, exactly as before this parameter existed --
    # proven by the absence of an explicit `timeout` kwarg on the call,
    # not by inspecting a numeric value that could coincidentally match
    # either path.
    body = read_fixture_text("node_arcs_many_parse.json")
    transport, _ = _replay(body)
    reader = _reader(transport)
    calls: list[dict] = []
    original_post = reader._client.post

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original_post(*args, **kwargs)

    monkeypatch.setattr(reader._client, "post", spy)

    reader.node_arcs_many(["country/KEN"], properties=("typeOf",))

    assert len(calls) == 1
    assert "timeout" not in calls[0]


def test_node_arcs_many_deadline_forwarded_unchanged_to_every_read_in_the_ladder(
    monkeypatch: pytest.MonkeyPatch,
):
    # The initial batch, the group retry, and each individual probe all
    # take the same absolute deadline -- proven here by advancing a
    # controlled clock between each of the four reads this scenario
    # triggers and checking that every request's own httpx timeout
    # shrinks accordingly, rather than resetting to a fresh full budget.
    first_body = json.dumps({"data": {"country/AAA": {}, "country/BBB": {}}})
    group_retry_body = json.dumps({"data": {"country/AAA": {}, "country/BBB": {}}})
    aaa_probe_body = json.dumps(
        {"data": {"country/AAA": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    bbb_probe_body = json.dumps(
        {"data": {"country/BBB": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    responses = [first_body, group_retry_body, aaa_probe_body, bbb_probe_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler), timeout=1000)

    # Each read now calls time.monotonic() twice: once to size the
    # node-arc semaphore acquire's own timeout (_acquire_node_arc_slot),
    # once for post's own request timeout. The clock is unheld here (no
    # concurrent reader), so the acquire always succeeds immediately
    # regardless of its computed timeout; each value is repeated so the
    # request timeout this test asserts on is unaffected by the extra call.
    monotonic_values = iter([0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 25.0, 25.0])
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: next(monotonic_values))

    reader.node_arcs_many(["country/AAA", "country/BBB"], properties=("typeOf",), deadline=100.0)

    assert len(requests) == 4
    assert [r.extensions["timeout"]["read"] for r in requests] == [100, 90, 80, 75]


def test_node_arcs_many_single_property_uses_unbracketed_arc_spec():
    body = read_fixture_text("node_arcs_many_parse.json")
    transport, requests = _replay(body)
    reader = _reader(transport)

    reader.node_arcs_many(["country/KEN"], properties=("typeOf",))

    payload = json.loads(requests[0].content)
    assert payload["property"] == "->typeOf"


def _names_body(dcid: str, *, name: Sequence[str] = (), alternate: Sequence[str] = ()) -> dict:
    """A `/v2/node` response naming `name`/`alternate` as `dcid`'s own
    `name` and `alternateName` out-arc entries."""
    arcs: dict = {}
    if name:
        arcs["name"] = {"nodes": [{"value": v} for v in name]}
    if alternate:
        arcs["alternateName"] = {"nodes": [{"value": v} for v in alternate]}
    return {"data": {dcid: {"arcs": arcs}}}


def test_node_names_many_requests_both_arcs_and_merges_in_order():
    body = json.dumps(_names_body("country/KEN", name=["Kenya"], alternate=["Republic of Kenya"]))
    transport, requests = _replay(body)
    reader = _reader(transport)

    result = reader.node_names_many(["country/KEN"])

    assert result == {"country/KEN": ("Kenya", "Republic of Kenya")}
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload == {"nodes": ["country/KEN"], "property": "->[name,alternateName]"}


def test_node_names_many_collapses_a_value_declared_by_a_doubled_provenance():
    # A redeclaring import contributes the same name value twice on the same
    # arc, and once more on the other arc. Both collapse to one entry, in
    # first-occurrence order.
    body = json.dumps(
        _names_body("country/KEN", name=["Kenya", "Kenya"], alternate=["Kenya", "Republic"])
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.node_names_many(["country/KEN"])

    assert result == {"country/KEN": ("Kenya", "Republic")}


def test_node_names_many_unknown_dcid_and_nameless_node_both_map_to_empty():
    body = json.dumps(
        {"data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.node_names_many(["country/KEN"])

    assert result == {"country/KEN": ()}


def test_node_names_many_duplicate_requested_dcids_collapse_to_one_key():
    body = json.dumps(_names_body("country/KEN", name=["Kenya"]))
    transport, requests = _replay(body)
    reader = _reader(transport)

    result = reader.node_names_many(["country/KEN", "country/KEN"])

    assert result == {"country/KEN": ("Kenya",)}
    assert len(requests) == 1


def test_node_names_many_empty_dcids_returns_empty_dict_no_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("node_names_many must not call the transport for empty dcids")

    reader = _reader(httpx.MockTransport(handler))

    assert reader.node_names_many([]) == {}


def test_node_names_many_transport_error_raises_graph_read_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphReadError):
        reader.node_names_many(["country/KEN"])


def test_node_names_many_drops_the_unresolved_target_marker_beside_a_real_name():
    # `node_arcs_many` marks an arc entry carrying neither `dcid` nor
    # `value` with `UNRESOLVED_NODE`; that entry carries no name, so only
    # the real value is reported.
    body = json.dumps(
        {
            "data": {
                "country/KEN": {"arcs": {"name": {"nodes": [{"value": "Kenya"}, {"foo": "bar"}]}}}
            }
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.node_names_many(["country/KEN"])

    assert result == {"country/KEN": ("Kenya",)}
    assert UNRESOLVED_NODE not in result["country/KEN"]


def test_node_names_many_maps_a_name_arc_of_only_the_unresolved_target_marker_to_empty():
    body = json.dumps({"data": {"country/KEN": {"arcs": {"name": {"nodes": [{"foo": "bar"}]}}}}})
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.node_names_many(["country/KEN"])

    assert result == {"country/KEN": ()}


@pytest.mark.parametrize("bad_char", [",", ":", "{", "}", "<", ">", "[", "]", " ", "\t"])
def test_arc_spec_rejects_unsafe_char_single_property(bad_char: str):
    with pytest.raises(ValueError):
        _arc_spec((f"bad{bad_char}prop",))


@pytest.mark.parametrize("bad_char", [",", ":", "{", "}", "<", ">", "[", "]", " ", "\t"])
def test_arc_spec_rejects_unsafe_char_multi_property(bad_char: str):
    with pytest.raises(ValueError):
        _arc_spec(("typeOf", f"bad{bad_char}prop"))


def test_arc_spec_legitimate_single_property_unchanged():
    assert _arc_spec(("typeOf",)) == "->typeOf"


def test_arc_spec_legitimate_multi_property_unchanged():
    assert _arc_spec(("typeOf", "name", "description")) == "->[typeOf,name,description]"


def test_node_arcs_drops_valueless_entry():
    # Single-node reads drop valueless targets; batched reads mark them `UNRESOLVED_NODE`.
    body = json.dumps(
        {
            "data": {
                "country/KEN": {
                    "arcs": {"measurementDenominator": {"nodes": [{"dcid": "Count_Person"}, {}]}}
                }
            }
        }
    )
    transport, _requests = _replay(body)
    reader = _reader(transport)

    result = reader.node_arc_values("country/KEN", properties=("measurementDenominator",))

    assert result == {"measurementDenominator": ("Count_Person",)}


def test_node_arcs_dedups_a_value_declared_by_two_provenances():
    # (#698) `node_arcs` collapses a shape value duplicated across provenances.
    body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {
                        "measuredProperty": {
                            "nodes": [
                                {"dcid": "lifeExpectancy"},
                                {"dcid": "lifeExpectancy"},
                            ]
                        }
                    }
                }
            }
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.node_arc_values("lifeExpectancy", properties=("measuredProperty",))

    assert result == {"measuredProperty": ("lifeExpectancy",)}


def test_node_arcs_many_dedups_a_value_declared_by_two_provenances():
    body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {
                        "measuredProperty": {
                            "nodes": [
                                {"dcid": "lifeExpectancy"},
                                {"dcid": "lifeExpectancy"},
                            ]
                        }
                    }
                }
            }
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.node_arcs_many(["lifeExpectancy"], properties=("measuredProperty",))

    assert result == {"lifeExpectancy": {"measuredProperty": ("lifeExpectancy",)}}


def test_node_arcs_many_dedups_unresolved_node_marker():
    # Duplicate unresolved targets collapse to one marker.
    body = json.dumps({"data": {"sv": {"arcs": {"measurementDenominator": {"nodes": [{}, {}]}}}}})
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.node_arcs_many(["sv"], properties=("measurementDenominator",))

    assert result == {"sv": {"measurementDenominator": (UNRESOLVED_NODE,)}}


def test_node_arcs_rejects_hostile_property_before_any_request():
    transport, requests = _replay("{}")
    reader = _reader(transport)

    with pytest.raises(ValueError):
        reader.node_arc_values("country/KEN", properties=("foo],<-typeOf{value:Property",))
    assert requests == []


def test_node_arcs_many_rejects_hostile_property_before_any_request():
    transport, requests = _replay("{}")
    reader = _reader(transport)

    with pytest.raises(ValueError):
        reader.node_arcs_many(
            ["country/KEN"], properties=("typeOf", "foo],<-typeOf{value:Property")
        )
    assert requests == []


def test_node_arcs_many_empty_dcids_returns_empty_dict_no_call():
    transport, requests = _replay("{}")
    reader = _reader(transport)

    assert reader.node_arcs_many([], properties=("typeOf",)) == {}
    assert requests == []


def test_node_arcs_many_retries_only_dropped_dcids_once():
    # The singleton retry recovers `country/TGO`, omitted from the initial batch.
    first_body = json.dumps(
        {
            "data": {
                "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
            }
        }
    )
    second_body = json.dumps(
        {
            "data": {
                "country/TGO": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
            }
        }
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.node_arcs_many(["country/KEN", "country/TGO"], properties=("typeOf",))

    assert len(requests) == 2
    second_payload = json.loads(requests[1].content)
    assert second_payload["nodes"] == ["country/TGO"]
    assert result == {
        "country/KEN": {"typeOf": ("Country",)},
        "country/TGO": {"typeOf": ("Country",)},
    }


def test_node_arcs_many_still_missing_after_retry_raises_graph_read_error():
    # Persistent omission after retry is a protocol violation.
    body = json.dumps(
        {"data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    transport, requests = _replay(body)
    reader = _reader(transport)

    with pytest.raises(GraphReadError):
        reader.node_arcs_many(["country/KEN", "country/TGO"], properties=("typeOf",))
    assert len(requests) == 2


def test_node_subjects_bare_request_shape():
    body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {
                        "measuredProperty": {
                            "nodes": [{"dcid": "dc/statvar1"}, {"dcid": "dc/statvar2"}]
                        }
                    }
                }
            }
        }
    )
    transport, requests = _replay(body)
    reader = _reader(transport)

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert page == SubjectPage(dcids=("dc/statvar1", "dc/statvar2"), truncated=False)
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload == {"nodes": ["lifeExpectancy"], "property": "<-measuredProperty"}


def test_node_subjects_filtered_request_shape_sorts_keys():
    body = json.dumps({"data": {"lifeExpectancy": {"arcs": {}}}})
    transport, requests = _replay(body)
    reader = _reader(transport)

    reader.node_subjects(
        "lifeExpectancy",
        property="measuredProperty",
        filters={"statType": "measuredValue", "populationType": "Person"},
        max_results=10,
    )

    payload = json.loads(requests[0].content)
    assert payload["property"] == "<-measuredProperty{populationType:Person,statType:measuredValue}"


def test_node_subjects_pages_via_next_token_until_exhausted():
    first_body = json.dumps(
        {
            "data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}]}}}},
            "nextToken": "tok1",
        }
    )
    second_body = json.dumps(
        {"data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "b"}]}}}}}
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert page == SubjectPage(dcids=("a", "b"), truncated=False)
    assert len(requests) == 2
    second_payload = json.loads(requests[1].content)
    assert second_payload == {
        "nodes": ["lifeExpectancy"],
        "property": "<-measuredProperty",
        "nextToken": "tok1",
    }


def test_node_subjects_truncated_true_when_continuation_page_has_new_distinct_subject():
    # A new distinct dcid on the continuation page proves truncation.
    first_body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}, {"dcid": "b"}]}}
                }
            },
            "nextToken": "tok1",
        }
    )
    second_body = json.dumps(
        {"data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "c"}]}}}}}
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=2)

    assert page == SubjectPage(dcids=("a", "b"), truncated=True)
    assert len(requests) == 2


def test_node_subjects_not_truncated_when_continuation_page_has_only_duplicates():
    # (#698) Duplicate-only continuation pages do not mark truncation.
    first_body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}, {"dcid": "b"}]}}
                }
            },
            "nextToken": "tok1",
        }
    )
    second_body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}, {"dcid": "b"}]}}
                }
            }
        }
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=2)

    assert page == SubjectPage(dcids=("a", "b"), truncated=False)
    assert len(requests) == 2


def test_node_subjects_fewer_than_max_results_not_truncated_on_natural_end():
    # Natural exhaustion below `max_results` is complete despite earlier tokens.
    first_body = json.dumps(
        {
            "data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}]}}}},
            "nextToken": "tok1",
        }
    )
    second_body = json.dumps(
        {"data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "b"}]}}}}}
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert page == SubjectPage(dcids=("a", "b"), truncated=False)
    assert len(requests) == 2


def test_node_subjects_truncated_false_on_natural_exhaustion():
    body = json.dumps(
        {"data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}]}}}}}
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert page == SubjectPage(dcids=("a",), truncated=False)


def test_node_subjects_skips_entries_with_no_dcid():
    body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {
                        "measuredProperty": {"nodes": [{"dcid": "a"}, {"value": "not a dcid"}, {}]}
                    }
                }
            }
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert page == SubjectPage(dcids=("a",), truncated=False)


def test_node_subjects_dedups_a_subject_declared_by_two_provenances():
    # (#698) Duplicate subject arcs collapse to one distinct StatVar.
    body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {
                        "measuredProperty": {
                            "nodes": [
                                {"dcid": "LifeExpectancy_Person_Female"},
                                {"dcid": "LifeExpectancy_Person_Female"},
                            ]
                        }
                    }
                }
            }
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert page == SubjectPage(dcids=("LifeExpectancy_Person_Female",), truncated=False)


def test_node_subjects_dedups_a_subject_repeated_across_pages():
    first_body = json.dumps(
        {
            "data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}]}}}},
            "nextToken": "tok1",
        }
    )
    second_body = json.dumps(
        {
            "data": {
                "lifeExpectancy": {
                    "arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}, {"dcid": "b"}]}}
                }
            }
        }
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert page == SubjectPage(dcids=("a", "b"), truncated=False)
    assert len(requests) == 2


def test_node_subjects_max_results_and_truncated_count_distinct_subjects():
    # 1,200 raw entries collapse to 600 distinct subjects, below the 1,000-result cap.
    # Counting raw entries would stop at 1,000 and could report truncation.
    distinct = [f"sv/{i}" for i in range(600)]
    doubled_nodes = [{"dcid": dcid} for dcid in distinct for _ in range(2)]
    body = json.dumps(
        {"data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": doubled_nodes}}}}}
    )
    transport, requests = _replay(body)
    reader = _reader(transport)

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=1000)

    assert page.dcids == tuple(distinct)
    assert page.truncated is False
    assert len(requests) == 1


def test_node_subjects_transport_error_raises_graph_read_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphReadError):
        reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)


def test_node_subjects_unescapable_filter_key_raises_value_error():
    reader = _reader(httpx.MockTransport(lambda request: httpx.Response(200, content="{}")))

    with pytest.raises(ValueError):
        reader.node_subjects(
            "lifeExpectancy",
            property="measuredProperty",
            filters={"bad,key": "value"},
            max_results=10,
        )


def test_node_subjects_unescapable_filter_value_raises_value_error():
    reader = _reader(httpx.MockTransport(lambda request: httpx.Response(200, content="{}")))

    with pytest.raises(ValueError):
        reader.node_subjects(
            "lifeExpectancy",
            property="measuredProperty",
            filters={"statType": "bad value"},
            max_results=10,
        )


def test_node_subjects_unescapable_property_raises_value_error():
    reader = _reader(httpx.MockTransport(lambda request: httpx.Response(200, content="{}")))

    with pytest.raises(ValueError):
        reader.node_subjects("lifeExpectancy", property="bad,property", max_results=10)


def test_node_subjects_repeated_next_token_raises_graph_not_advancing_error():
    # Repeating `tok1` raises after two pages; a third read exhausts the fake queue.
    page_body = json.dumps(
        {
            "data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}]}}}},
            "nextToken": "tok1",
        }
    )
    responses = [page_body, page_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphNotAdvancingError) as excinfo:
        reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=100)
    assert len(requests) == 2
    assert excinfo.value.dcid == "lifeExpectancy"
    assert excinfo.value.property == "measuredProperty"


def _subject_page_body(dcid_values: list[str], *, next_token: str | None = None) -> str:
    """Build a `node_subjects` page for the fixture StatVar and property."""
    body: dict = {
        "data": {
            "lifeExpectancy": {
                "arcs": {"measuredProperty": {"nodes": [{"dcid": d} for d in dcid_values]}}
            }
        }
    }
    if next_token is not None:
        body["nextToken"] = next_token
    return json.dumps(body)


def test_node_subjects_single_sparse_page_does_not_raise_and_still_completes():
    # (#697) Provenance-keyed pages may contain duplicates before a later page adds a new subject.
    responses = [
        _subject_page_body(["a"], next_token="tok1"),
        _subject_page_body(["a"], next_token="tok2"),  # only a duplicate; not proof of stuck
        _subject_page_body(["b"]),  # new dcid, no token: reads complete
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=100)

    assert page == SubjectPage(dcids=("a", "b"), truncated=False)
    assert len(requests) == 3


def test_node_subjects_sustained_no_progress_raises_graph_not_advancing_error():
    # (#697) Fresh tokens do not prove progress; repeated subjects hit the consecutive-page limit.
    limit = graph_module._NODE_SUBJECTS_STUCK_PAGE_LIMIT
    responses = [_subject_page_body(["a"], next_token="tok0")]
    responses += [_subject_page_body(["a"], next_token=f"tok{i}") for i in range(1, limit + 1)]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphNotAdvancingError) as excinfo:
        reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=100)
    assert len(requests) == 1 + limit
    assert excinfo.value.dcid == "lifeExpectancy"
    assert excinfo.value.property == "measuredProperty"


def test_node_subjects_sustained_zero_dcids_raises_graph_not_advancing_error():
    # Empty pages count toward the limit from the first request.
    limit = graph_module._NODE_SUBJECTS_STUCK_PAGE_LIMIT
    responses = [_subject_page_body([], next_token=f"tok{i}") for i in range(limit)]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphNotAdvancingError):
        reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=100)
    assert len(requests) == limit


def test_node_subjects_except_graph_read_error_catches_both_not_advancing_signals():
    # GraphNotAdvancingError subclasses GraphReadError, so an existing
    # `except GraphReadError` handler keeps catching it unchanged, whether
    # it came from a repeated token or sustained no-progress on fresh ones.
    limit = graph_module._NODE_SUBJECTS_STUCK_PAGE_LIMIT
    repeated_token_body = _subject_page_body(["a"], next_token="tok1")
    fresh_token_no_progress = [_subject_page_body([], next_token=f"tok{i}") for i in range(limit)]

    def repeated_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=repeated_token_body)

    caught = []
    try:
        _reader(httpx.MockTransport(repeated_handler)).node_subjects(
            "lifeExpectancy", property="measuredProperty", max_results=100
        )
    except GraphReadError as exc:
        caught.append(exc)

    requests: list[httpx.Request] = []

    def fresh_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=fresh_token_no_progress[len(requests) - 1])

    try:
        _reader(httpx.MockTransport(fresh_handler)).node_subjects(
            "lifeExpectancy", property="measuredProperty", max_results=100
        )
    except GraphReadError as exc:
        caught.append(exc)

    assert len(caught) == 2
    assert all(isinstance(exc, GraphNotAdvancingError) for exc in caught)


def test_node_subjects_page_budget_exhausted_with_real_progress_raises_graph_read_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # New dcids on both pages avoid the stuck-cursor error; the page bound raises `GraphReadError`.
    monkeypatch.setattr(graph_module, "_NODE_SUBJECTS_MAX_PAGES", 2)
    responses = [
        json.dumps(
            {
                "data": {
                    "lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}]}}}
                },
                "nextToken": "tok1",
            }
        ),
        json.dumps(
            {
                "data": {
                    "lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "b"}]}}}
                },
                "nextToken": "tok2",
            }
        ),
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    with pytest.raises(GraphReadError) as excinfo:
        reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=100)
    assert not isinstance(excinfo.value, GraphNotAdvancingError)
    assert len(requests) == 2


def test_node_subjects_multi_page_walk_unchanged_by_the_page_bound():
    # Three productive pages exhaust normally with `truncated=False`.
    responses = [
        json.dumps(
            {
                "data": {
                    "lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}]}}}
                },
                "nextToken": "tok1",
            }
        ),
        json.dumps(
            {
                "data": {
                    "lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "b"}]}}}
                },
                "nextToken": "tok2",
            }
        ),
        json.dumps(
            {"data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "c"}]}}}}}
        ),
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    page = reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert page == SubjectPage(dcids=("a", "b", "c"), truncated=False)
    assert len(requests) == 3


def test_node_subjects_expired_deadline_raises_before_any_request():
    transport = _RecordingTransport()
    reader = _reader(transport)

    with pytest.raises(GraphDeadlineExceededError, match="absolute caller deadline"):
        reader.node_subjects(
            "lifeExpectancy",
            property="measuredProperty",
            max_results=10,
            deadline=time.monotonic() - 1,
        )

    assert transport.request is None


def test_node_subjects_deadline_none_behaves_exactly_as_before(monkeypatch: pytest.MonkeyPatch):
    # No `deadline` given issues every page through the client's own
    # configured timeout, exactly as before this parameter existed --
    # proven by the absence of an explicit `timeout` kwarg on the call,
    # not by inspecting a numeric value that could coincidentally match
    # either path.
    body = json.dumps({"data": {"lifeExpectancy": {"arcs": {}}}})
    transport, _ = _replay(body)
    reader = _reader(transport)
    calls: list[dict] = []
    original_post = reader._client.post

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original_post(*args, **kwargs)

    monkeypatch.setattr(reader._client, "post", spy)

    reader.node_subjects("lifeExpectancy", property="measuredProperty", max_results=10)

    assert len(calls) == 1
    assert "timeout" not in calls[0]


def test_node_subjects_paging_shares_one_absolute_deadline_across_pages(
    monkeypatch: pytest.MonkeyPatch,
):
    # Two pages, one absolute deadline: the controlled clock advances by
    # 30 between the two `post` calls this triggers, simulating time
    # actually spent on the first page's own network round trip. A fresh
    # per-page budget would hand the second page the full configured
    # timeout again; sharing one absolute deadline instead leaves it with
    # only what's left.
    first_body = json.dumps(
        {
            "data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "a"}]}}}},
            "nextToken": "tok1",
        }
    )
    second_body = json.dumps(
        {"data": {"lifeExpectancy": {"arcs": {"measuredProperty": {"nodes": [{"dcid": "b"}]}}}}}
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler), timeout=1000)

    monotonic_values = iter([10.0, 40.0])
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: next(monotonic_values))

    reader.node_subjects(
        "lifeExpectancy", property="measuredProperty", max_results=10, deadline=50.0
    )

    assert len(requests) == 2
    assert requests[0].extensions["timeout"]["read"] == 40
    assert requests[1].extensions["timeout"]["read"] == 10


def test_child_places_batch_issues_one_request_per_parent():
    # The dev graph's mixer times out computing this transitive-containment
    # closure whenever two or more non-trivial parents share one request,
    # even though each parent alone is fast (verified live). So this always
    # issues one request per parent, never a batched request across parents.
    responses = [
        json.dumps(
            {
                "data": {
                    "africa": {
                        "arcs": {
                            "containedInPlace+": {
                                "nodes": [{"dcid": "country/KEN"}, {"dcid": "country/NGA"}]
                            }
                        }
                    }
                }
            }
        ),
        json.dumps(
            {
                "data": {
                    "SubSaharanAfrica": {
                        "arcs": {
                            "containedInPlace+": {
                                "nodes": [{"dcid": "country/KEN"}, {"dcid": "country/ZAF"}]
                            }
                        }
                    }
                }
            }
        ),
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.child_places_batch(["africa", "SubSaharanAfrica"], child_type="Country")

    assert len(requests) == 2
    for request in requests:
        assert request.method == "POST"
        assert request.url.path == "/core/api/v2/node"
    payloads = [json.loads(r.content) for r in requests]
    assert payloads[0] == {"nodes": ["africa"], "property": "<-containedInPlace+{typeOf:Country}"}
    assert payloads[1] == {
        "nodes": ["SubSaharanAfrica"],
        "property": "<-containedInPlace+{typeOf:Country}",
    }
    assert result == {
        "africa": ["country/KEN", "country/NGA"],
        "SubSaharanAfrica": ["country/KEN", "country/ZAF"],
    }


def test_child_places_batch_empty_parent_list_makes_no_call():
    transport, requests = _replay("{}")
    reader = _reader(transport)

    assert reader.child_places_batch([], child_type="Country") == {}
    assert requests == []


def test_child_places_batch_expired_deadline_raises_before_any_request():
    # The deadline reaches each parent's own page POST through `post`:
    # a budget already spent raises there, before the wire.
    transport, requests = _replay("{}")
    reader = _reader(transport)

    with pytest.raises(GraphDeadlineExceededError, match="absolute caller deadline"):
        reader.child_places_batch(["africa"], child_type="Country", deadline=time.monotonic() - 1)

    assert requests == []


def test_child_places_batch_deadline_bounds_every_parents_reads():
    # One absolute deadline covers the whole batch: with five seconds left
    # of it, each parent's own POST takes that remaining budget rather than
    # the reader's configured timeout (each parent gets its own POST, see
    # `child_places_batch`'s own docstring).
    transport = _RecordingTransport()
    reader = _reader(transport, timeout=90)

    reader.child_places_batch(
        ["africa", "SubSaharanAfrica"], child_type="Country", deadline=time.monotonic() + 5
    )

    assert transport.request is not None
    assert 0 < transport.request.extensions["timeout"]["read"] <= 5


def test_child_places_batch_timeout_under_a_deadline_raises_deadline_exceeded():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("transport gave up", request=request)

    reader = _reader(httpx.MockTransport(handler), timeout=30)

    with pytest.raises(GraphDeadlineExceededError):
        reader.child_places_batch(["africa"], child_type="Country", deadline=time.monotonic() + 30)


def test_child_places_batch_pages_each_parent_independently():
    # Parent "a" needs two pages, parent "b" needs one. Each parent's own
    # paging walk is independent -- "b"'s single-page response does not
    # affect how many pages "a" is read for.
    responses = [
        json.dumps(
            {
                "data": {"a": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "child/1"}]}}}},
                "nextToken": "tok1",
            }
        ),
        json.dumps(
            {"data": {"a": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "child/2"}]}}}}}
        ),
        json.dumps(
            {"data": {"b": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "child/3"}]}}}}}
        ),
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.child_places_batch(["a", "b"], child_type="Country")

    assert len(requests) == 3
    assert json.loads(requests[0].content) == {
        "nodes": ["a"],
        "property": "<-containedInPlace+{typeOf:Country}",
    }
    assert json.loads(requests[1].content) == {
        "nodes": ["a"],
        "property": "<-containedInPlace+{typeOf:Country}",
        "nextToken": "tok1",
    }
    assert json.loads(requests[2].content) == {
        "nodes": ["b"],
        "property": "<-containedInPlace+{typeOf:Country}",
    }
    assert result == {"a": ["child/1", "child/2"], "b": ["child/3"]}


def test_child_places_batch_dedups_a_child_declared_by_two_provenances():
    # (#698) A child place asserted via ->containedInPlace by two provenances
    # reaches this read as two arc entries for the same dcid; a genuinely
    # distinct sibling survives once, undisturbed by the dedup.
    body = json.dumps(
        {
            "data": {
                "africa": {
                    "arcs": {
                        "containedInPlace+": {
                            "nodes": [
                                {"dcid": "country/KEN"},
                                {"dcid": "country/KEN"},
                                {"dcid": "country/NGA"},
                            ]
                        }
                    }
                }
            }
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.child_places_batch(["africa"], child_type="Country")

    assert result == {"africa": ["country/KEN", "country/NGA"]}


def test_child_places_batch_dedups_a_child_repeated_across_pages():
    first_body = json.dumps(
        {
            "data": {
                "africa": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/KEN"}]}}}
            },
            "nextToken": "tok1",
        }
    )
    second_body = json.dumps(
        {
            "data": {
                "africa": {
                    "arcs": {
                        "containedInPlace+": {
                            "nodes": [{"dcid": "country/KEN"}, {"dcid": "country/NGA"}]
                        }
                    }
                }
            }
        }
    )
    responses = [first_body, second_body]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    reader = _reader(httpx.MockTransport(handler))

    result = reader.child_places_batch(["africa"], child_type="Country")

    assert result == {"africa": ["country/KEN", "country/NGA"]}
    assert len(requests) == 2


def test_resolve_by_arc_maps_every_name_preserves_order_empty_list_for_no_candidates():
    body = json.dumps(
        {
            "entities": [
                {
                    "node": "Kenya",
                    "candidates": [{"dcid": "country/KEN"}, {"dcid": "country/KEN2"}],
                },
                {"node": "Nowhereland", "candidates": []},
            ]
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.resolve_by_arc(["Kenya", "Nowhereland"], arc="<-name->dcid")

    assert result == {"Kenya": ["country/KEN", "country/KEN2"], "Nowhereland": []}


def test_resolve_by_arc_omitted_name_maps_to_empty_list_unrequested_node_ignored():
    body = json.dumps(
        {
            "entities": [
                {"node": "Togo", "candidates": [{"dcid": "country/TGO"}]},
                {"node": "Unrequested", "candidates": [{"dcid": "country/ZZZ"}]},
            ]
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.resolve_by_arc(["Togo", "Kenya"], arc="<-name->dcid")

    assert result == {"Togo": ["country/TGO"], "Kenya": []}


def test_resolve_by_arc_candidate_without_dcid_is_skipped():
    body = json.dumps(
        {
            "entities": [
                {
                    "node": "Kenya",
                    "candidates": [{"dcid": "country/KEN"}, {"dominantType": "Country"}],
                }
            ]
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.resolve_by_arc(["Kenya"], arc="<-name->dcid")

    assert result == {"Kenya": ["country/KEN"]}


def test_resolve_by_arc_dedups_a_candidate_declared_by_two_provenances():
    # (#698) A candidate declared by two Data Commons provenances reports
    # the same dcid twice among a name's candidates. resolve_by_arc
    # collapses the duplicate.
    body = json.dumps(
        {
            "entities": [
                {
                    "node": "Kenya",
                    "candidates": [{"dcid": "country/KEN"}, {"dcid": "country/KEN"}],
                }
            ]
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.resolve_by_arc(["Kenya"], arc="<-name->dcid")

    assert result == {"Kenya": ["country/KEN"]}


def test_resolve_by_arc_keeps_genuinely_distinct_candidates():
    body = json.dumps(
        {
            "entities": [
                {
                    "node": "Congo",
                    "candidates": [{"dcid": "country/COD"}, {"dcid": "country/COG"}],
                }
            ]
        }
    )
    transport, _ = _replay(body)
    reader = _reader(transport)

    result = reader.resolve_by_arc(["Congo"], arc="<-name->dcid")

    assert result == {"Congo": ["country/COD", "country/COG"]}


def test_resolve_by_arc_empty_names_returns_empty_dict_no_request():
    transport, requests = _replay(json.dumps({"entities": []}))
    reader = _reader(transport)

    assert reader.resolve_by_arc([], arc="<-name->dcid") == {}
    assert len(requests) == 0


def test_resolve_by_arc_request_shape_single_post():
    body = json.dumps({"entities": []})
    transport, requests = _replay(body)
    reader = _reader(transport)

    reader.resolve_by_arc(["Kenya", "Togo"], arc="<-name->dcid")

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/core/api/v2/resolve"
    payload = json.loads(request.content)
    assert payload == {"nodes": ["Kenya", "Togo"], "property": "<-name->dcid"}


def test_post_retries_500_then_succeeds_two_requests_one_backoff(monkeypatch):
    ok_body = read_fixture_text("node_label_found.json")
    transport, requests = _sequenced_transport(
        [
            httpx.Response(500, content="boom", headers={"content-type": "text/plain"}),
            httpx.Response(200, content=ok_body, headers={"content-type": "application/json"}),
        ]
    )
    reader = _reader(transport)
    sleeps: list[float] = []
    monkeypatch.setattr(graph_module.time, "sleep", sleeps.append)

    label = reader.node_label("country/KEN")

    assert label == "Kenya"
    assert len(requests) == 2
    assert sleeps == [graph_module._RETRY_BACKOFF_SECONDS[0]]


def test_post_retries_500_on_every_attempt_raises_after_max_attempts(monkeypatch):
    transport, requests = _sequenced_transport(
        [
            httpx.Response(500, content="boom", headers={"content-type": "text/plain"}),
            httpx.Response(500, content="boom", headers={"content-type": "text/plain"}),
            httpx.Response(500, content="boom", headers={"content-type": "text/plain"}),
        ]
    )
    reader = _reader(transport)  # default max_attempts=3
    sleeps: list[float] = []
    monkeypatch.setattr(graph_module.time, "sleep", sleeps.append)

    with pytest.raises(GraphReadError) as excinfo:
        reader.node_label("country/KEN")

    assert excinfo.value.upstream_status == 500
    assert len(requests) == 3
    assert sleeps == list(graph_module._RETRY_BACKOFF_SECONDS)


def test_post_never_retries_a_404():
    transport, requests = _sequenced_transport(
        [httpx.Response(404, content="not found", headers={"content-type": "text/plain"})]
    )
    reader = _reader(transport)

    with pytest.raises(GraphReadError) as excinfo:
        reader.node_label("country/KEN")

    assert excinfo.value.upstream_status == 404
    assert len(requests) == 1


def test_post_retries_request_error_then_succeeds_two_requests():
    ok_body = read_fixture_text("node_label_found.json")
    transport, requests = _sequenced_transport(
        [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, content=ok_body, headers={"content-type": "application/json"}),
        ]
    )
    reader = _reader(transport)

    label = reader.node_label("country/KEN")

    assert label == "Kenya"
    assert len(requests) == 2


def test_post_retries_a_connect_error_under_a_deadline(monkeypatch):
    # A connection failure is as transient as a 5xx, so a deadline caller
    # gets the same retry a deadline-free one does, bounded by the same
    # budget check. A timeout under a deadline is only different when the
    # deadline itself bounded the attempt, or has already passed: only
    # then has the timeout spent the budget the caller set, so it raises
    # instead.
    ok_body = read_fixture_text("observation_facets_found.json")
    transport, requests = _sequenced_transport(
        [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, content=ok_body, headers={"content-type": "application/json"}),
        ]
    )
    reader = _reader(transport)
    sleeps: list[float] = []
    monkeypatch.setattr(graph_module.time, "sleep", sleeps.append)

    reader.variable_facets_batch(
        stat_var="dcid:Whatever",
        entities=["country/KEN"],
        deadline=time.monotonic() + 30.0,
    )

    assert len(requests) == 2
    assert sleeps == [graph_module._RETRY_BACKOFF_SECONDS[0]]


def test_timeout_bounded_by_reader_timeout_not_deadline_is_retried():
    # The reader's own configured timeout (2s) bounds the first attempt's
    # per-call timeout (`min(self._timeout, remaining)` picks 2), well
    # under the deadline's 60s remaining budget. A timeout there is an
    # ordinary transient failure, retried through `_await_retry` like any
    # other `httpx.RequestError`.
    ok_body = read_fixture_text("observation_facets_found.json")
    transport, requests = _sequenced_transport(
        [
            httpx.ReadTimeout("simulated slow response"),
            httpx.Response(200, content=ok_body, headers={"content-type": "application/json"}),
        ]
    )
    reader = _reader(transport, timeout=2)

    reader.variable_facets_batch(
        stat_var="dcid:Whatever",
        entities=["country/KEN"],
        deadline=time.monotonic() + 60.0,
    )

    assert len(requests) == 2


def test_timeout_bounded_by_reader_timeout_exhausts_retries_raises_graph_read_error():
    # Every attempt times out on the reader's own 2s timeout, well under
    # the 60s deadline budget. Exhausting `max_attempts` this way raises
    # the plain `GraphReadError` any other exhausted retryable failure
    # would.
    transport, requests = _sequenced_transport(
        [
            httpx.ReadTimeout("simulated slow response"),
            httpx.ReadTimeout("simulated slow response"),
            httpx.ReadTimeout("simulated slow response"),
        ]
    )
    reader = _reader(transport, timeout=2)  # default max_attempts=3

    with pytest.raises(GraphReadError) as excinfo:
        reader.variable_facets_batch(
            stat_var="dcid:Whatever",
            entities=["country/KEN"],
            deadline=time.monotonic() + 60.0,
        )

    assert not isinstance(excinfo.value, GraphDeadlineExceededError)
    assert len(requests) == 3


def test_timeout_exception_after_deadline_has_passed_raises_graph_deadline_exceeded_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # The reader's own configured timeout (10s) bounds the attempt, well
    # under the deadline's 100s remaining budget at the top of the loop.
    # The deadline has elapsed by the time the timeout fires, so this is
    # a deadline miss. Exactly one request is made, since retrying stops
    # once the deadline is spent.
    transport, requests = _sequenced_transport([httpx.ReadTimeout("simulated slow response")])
    reader = _reader(transport, timeout=10)
    monotonic_values = iter([0.0, 150.0])
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(GraphDeadlineExceededError):
        reader.variable_facets_batch(
            stat_var="dcid:Whatever", entities=["country/KEN"], deadline=100.0
        )

    assert len(requests) == 1


def test_timeout_budget_too_small_for_backoff_raises_graph_read_error_after_one_request(
    monkeypatch: pytest.MonkeyPatch,
):
    # The reader's own configured timeout (10s) bounds the attempt, and
    # the deadline has not yet passed when the timeout fires, but so
    # little budget remains afterwards that even the next backoff (0.5s)
    # would run past the deadline. Following `_await_retry`'s contract,
    # retrying stops there and the timeout that triggered it is raised
    # as-is, the plain `GraphReadError` any other exhausted retryable
    # failure gets.
    transport, requests = _sequenced_transport([httpx.ReadTimeout("simulated slow response")])
    reader = _reader(transport, timeout=10)
    monotonic_values = iter([0.0, 50.0, 99.6])
    monkeypatch.setattr(graph_module.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(GraphReadError) as excinfo:
        reader.variable_facets_batch(
            stat_var="dcid:Whatever", entities=["country/KEN"], deadline=100.0
        )

    assert not isinstance(excinfo.value, GraphDeadlineExceededError)
    assert len(requests) == 1


def test_post_max_attempts_one_disables_retry_no_sleep(monkeypatch):
    transport, requests = _sequenced_transport(
        [httpx.Response(500, content="boom", headers={"content-type": "text/plain"})]
    )
    reader = _reader(transport, max_attempts=1)
    sleeps: list[float] = []
    monkeypatch.setattr(graph_module.time, "sleep", sleeps.append)

    with pytest.raises(GraphReadError) as excinfo:
        reader.node_label("country/KEN")

    assert excinfo.value.upstream_status == 500
    assert len(requests) == 1
    assert sleeps == []


def test_post_deadline_too_small_for_backoff_stops_retrying_without_sleeping(monkeypatch):
    # The first attempt fits comfortably inside the deadline and comes back
    # a retryable 500, but so little budget remains afterwards that even
    # the first backoff (0.5s) would run past the deadline. Retrying stops
    # right there instead of sleeping past it, and the 500 from that one
    # attempt is what gets raised.
    transport, requests = _sequenced_transport(
        [httpx.Response(500, content="boom", headers={"content-type": "text/plain"})]
    )
    reader = _reader(transport)
    sleeps: list[float] = []
    monkeypatch.setattr(graph_module.time, "sleep", sleeps.append)

    with pytest.raises(GraphReadError) as excinfo:
        reader.variable_facets_batch(
            stat_var="dcid:Whatever",
            entities=["country/KEN"],
            deadline=time.monotonic() + 0.1,
        )

    assert excinfo.value.upstream_status == 500
    assert len(requests) == 1
    assert sleeps == []


def test_post_retries_a_503_then_succeeds_with_no_explicit_max_attempts(monkeypatch):
    # GraphClient() with no max_attempts argument at all -- not _reader's own
    # explicit default of 3 -- retries a 503 the same way an explicit
    # max_attempts=3 does, proving the class's own constructor default.
    ok_body = read_fixture_text("node_label_found.json")
    transport, requests = _sequenced_transport(
        [
            httpx.Response(503, content="boom", headers={"content-type": "text/plain"}),
            httpx.Response(200, content=ok_body, headers={"content-type": "application/json"}),
        ]
    )
    reader = GraphClient(base_url="https://graph.example.org/core/api", transport=transport)
    sleeps: list[float] = []
    monkeypatch.setattr(graph_module.time, "sleep", sleeps.append)

    label = reader.node_label("country/KEN")

    assert label == "Kenya"
    assert len(requests) == 2
    assert sleeps == [graph_module._RETRY_BACKOFF_SECONDS[0]]


def test_post_max_attempts_zero_raises_value_error():
    with pytest.raises(ValueError, match="max_attempts"):
        GraphClient(
            base_url="https://graph.example.org/core/api",
            transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            max_attempts=0,
        )
