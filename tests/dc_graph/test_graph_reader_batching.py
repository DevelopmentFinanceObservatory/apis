"""`node_arcs_many` follows each `/v2/node` continuation token and merges its pages."""

from __future__ import annotations

import json

import httpx
import pytest

import dc_graph.client as graph_module
from dc_graph import GraphClient, GraphReadError


def _reader(transport: httpx.BaseTransport) -> GraphClient:
    return GraphClient(base_url="https://graph.example.org/core/api", transport=transport)


def _requests_handler(responses: list[str]) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    return httpx.MockTransport(handler), requests


def test_single_page_response_issues_exactly_one_request():
    body = json.dumps(
        {"data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    transport, requests = _requests_handler([body])
    reader = _reader(transport)

    result = reader.node_arcs_many(["country/KEN"], properties=("typeOf",))

    assert len(requests) == 1
    assert result == {"country/KEN": {"typeOf": ("Country",)}}


def test_paged_response_accumulates_across_pages():
    first_body = json.dumps(
        {
            "data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}},
            "nextToken": "page-2",
        }
    )
    second_body = json.dumps(
        {"data": {"country/TGO": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}
    )
    transport, requests = _requests_handler([first_body, second_body])
    reader = _reader(transport)

    result = reader.node_arcs_many(["country/KEN", "country/TGO"], properties=("typeOf",))

    assert len(requests) == 2
    assert json.loads(requests[0].content).get("nextToken") is None
    assert json.loads(requests[1].content)["nextToken"] == "page-2"
    assert result == {
        "country/KEN": {"typeOf": ("Country",)},
        "country/TGO": {"typeOf": ("Country",)},
    }


@pytest.mark.parametrize(
    ("pages", "properties", "expected_result"),
    [
        pytest.param(
            [
                json.dumps(
                    {
                        "data": {
                            "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}
                        },
                        "nextToken": "page-2",
                    }
                ),
                json.dumps(
                    {"data": {"country/KEN": {"arcs": {"name": {"nodes": [{"value": "Kenya"}]}}}}}
                ),
            ],
            ("typeOf", "name"),
            {"country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)}},
            id="distinct-properties",
        ),
        pytest.param(
            [
                json.dumps(
                    {
                        "data": {
                            "country/KEN": {
                                "arcs": {"containedInPlace": {"nodes": [{"dcid": "africa"}]}}
                            }
                        },
                        "nextToken": "page-2",
                    }
                ),
                json.dumps(
                    {
                        "data": {
                            "country/KEN": {
                                "arcs": {"containedInPlace": {"nodes": [{"dcid": "earth"}]}}
                            }
                        }
                    }
                ),
            ],
            ("containedInPlace",),
            {"country/KEN": {"containedInPlace": ("africa", "earth")}},
            id="same-property-prepends-next-page-values",
        ),
    ],
)
def test_paged_node_arcs_merge_without_clobbering_prior_page_values(
    pages: list[str],
    properties: tuple[str, ...],
    expected_result: dict[str, dict[str, tuple[str, ...]]],
):
    """Merge each page into the preceding page's node arcs."""
    transport, requests = _requests_handler(pages)
    reader = _reader(transport)

    result = reader.node_arcs_many(["country/KEN"], properties=properties)

    assert len(requests) == 2
    assert result == expected_result


def test_page_budget_exhausted_raises_graph_read_error(monkeypatch: pytest.MonkeyPatch):
    # Distinct tokens leave the page budget as the only stopping condition.
    monkeypatch.setattr(graph_module, "_NODE_ARCS_MANY_MAX_PAGES", 2)
    responses = [
        json.dumps(
            {
                "data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}},
                "nextToken": "keeps-going-1",
            }
        ),
        json.dumps(
            {
                "data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}},
                "nextToken": "keeps-going-2",
            }
        ),
    ]
    transport, requests = _requests_handler(responses)
    reader = _reader(transport)

    with pytest.raises(GraphReadError):
        reader.node_arcs_many(["country/KEN"], properties=("typeOf",))
    assert len(requests) == 2


def test_repeated_next_token_raises_graph_read_error_promptly():
    # Repeated `tok1` must fail after two pages; a third request exhausts the
    # fake response queue.
    page_body = json.dumps(
        {
            "data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}},
            "nextToken": "tok1",
        }
    )
    transport, requests = _requests_handler([page_body, page_body])
    reader = _reader(transport)

    with pytest.raises(GraphReadError):
        reader.node_arcs_many(["country/KEN"], properties=("typeOf",))
    assert len(requests) == 2


def test_empty_dcids_returns_empty_dict_no_call():
    transport, requests = _requests_handler([])
    reader = _reader(transport)

    assert reader.node_arcs_many([], properties=("typeOf",)) == {}
    assert requests == []


@pytest.mark.parametrize(
    ("responses", "dcids", "properties", "expected_result"),
    [
        pytest.param(
            [
                json.dumps(
                    {
                        "data": {
                            "country/KEN": {
                                "arcs": {
                                    "measurementDenominator": {
                                        "nodes": [{"dcid": "Count_Person"}, {}]
                                    }
                                }
                            }
                        }
                    }
                )
            ],
            ["country/KEN"],
            ("measurementDenominator",),
            {
                "country/KEN": {
                    "measurementDenominator": ("Count_Person", graph_module.UNRESOLVED_NODE)
                }
            },
            id="initial-batch",
        ),
        pytest.param(
            [
                json.dumps(
                    {
                        "data": {
                            "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                            "country/ZZZ": {},
                        }
                    }
                ),
                json.dumps({"data": {"country/ZZZ": {}}}),
                json.dumps({"data": {"country/ZZZ": {"arcs": {"typeOf": {"nodes": [{}]}}}}}),
            ],
            ["country/KEN", "country/ZZZ"],
            ("typeOf",),
            {
                "country/KEN": {"typeOf": ("Country",)},
                "country/ZZZ": {"typeOf": (graph_module.UNRESOLVED_NODE,)},
            },
            id="individual-probe",
        ),
    ],
)
def test_batched_paths_mark_a_valueless_entry_unresolved(
    responses: list[str],
    dcids: list[str],
    properties: tuple[str, ...],
    expected_result: dict[str, dict[str, tuple[str, ...]]],
):
    """Preserve the unresolved-marker contract across assessed batch paths."""
    transport, requests = _requests_handler(responses)
    reader = _reader(transport)

    result = reader.node_arcs_many(dcids, properties=properties)

    assert len(requests) == len(responses)
    assert result == expected_result


def test_group_retry_marks_a_valueless_entry_unresolved():
    # The retry returns an arc without `dcid` or `value`; keep it as `UNRESOLVED_NODE`.
    first_body = json.dumps(
        {
            "data": {
                "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                "country/ZZZ": {},
            }
        }
    )
    retry_body = json.dumps({"data": {"country/ZZZ": {"arcs": {"typeOf": {"nodes": [{}]}}}}})
    transport, requests = _requests_handler([first_body, retry_body])
    reader = _reader(transport)

    result = reader.node_arcs_many(["country/KEN", "country/ZZZ"], properties=("typeOf",))

    assert len(requests) == 2
    assert result["country/ZZZ"] == {"typeOf": (graph_module.UNRESOLVED_NODE,)}


def test_false_empty_group_retry_pages_independently_of_the_initial_batch():
    # The false-empty retry for `country/ZZZ` spans two pages and resolves the dcid.
    first_body = json.dumps(
        {
            "data": {
                "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                "country/ZZZ": {},
            }
        }
    )
    retry_page_1 = json.dumps(
        {
            "data": {"country/ZZZ": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}},
            "nextToken": "page-2",
        }
    )
    retry_page_2 = json.dumps(
        {"data": {"country/ZZZ": {"arcs": {"name": {"nodes": [{"value": "Unknown"}]}}}}}
    )
    transport, requests = _requests_handler([first_body, retry_page_1, retry_page_2])
    reader = _reader(transport)

    result = reader.node_arcs_many(["country/KEN", "country/ZZZ"], properties=("typeOf", "name"))

    assert len(requests) == 3
    assert result == {
        "country/KEN": {"typeOf": ("Country",), "name": ()},
        "country/ZZZ": {"typeOf": ("Country",), "name": ("Unknown",)},
    }
