"""Pins unresolved-arc classification and survey pagination with a stub graph."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import pytest

import scripts.survey_unresolved_arcs as m
from scripts.survey_unresolved_arcs import (
    classify_arc,
    format_report,
    reverse_probe,
    seed_sv_dcids,
    survey,
)
from search.eval.cases import validate_case

# ---------------------------------------------------------------------------
# classify_arc
# ---------------------------------------------------------------------------


def test_classify_arc_resolved_when_a_node_carries_a_dcid():
    arcs = {"measurementDenominator": {"nodes": [{"dcid": "Count_Person", "name": "x"}]}}
    assert classify_arc(arcs, "measurementDenominator") == "resolved"


def test_classify_arc_unresolved_when_nodes_present_without_dcid():
    """A dcid-less ProvisionalNode target: nodes present, none carries a dcid."""
    arcs = {"measurementDenominator": {"nodes": [{"name": "Per Capita", "types": ["Class"]}]}}
    assert classify_arc(arcs, "measurementDenominator") == "unresolved"


def test_classify_arc_absent_when_no_nodes():
    arcs = {"measurementDenominator": {"nodes": []}}
    assert classify_arc(arcs, "measurementDenominator") == "absent"


def test_classify_arc_absent_when_property_missing_entirely():
    arcs = {"otherProperty": {"nodes": [{"dcid": "x"}]}}
    assert classify_arc(arcs, "measurementDenominator") == "absent"


def test_classify_arc_resolved_when_only_some_nodes_carry_a_dcid():
    arcs = {
        "constraintProperties": {
            "nodes": [{"dcid": "activitySource"}, {"name": "unresolved sibling"}]
        }
    }
    assert classify_arc(arcs, "constraintProperties") == "resolved"


# ---------------------------------------------------------------------------
# seed_sv_dcids -- reads real corpus/fixture files, no network
# ---------------------------------------------------------------------------


def _case(case_id: str, expect: dict):
    return validate_case(
        {"id": case_id, "query": "aid to Kenya", "why": "A survey seed.", "expect": expect},
        families={},
    )


def test_seed_sv_dcids_reads_stat_vars_from_every_part_and_fixture_statvar(tmp_path, monkeypatch):
    fixture_nodes_path = tmp_path / "graph_nodes.json"
    fixture_nodes_path.write_text(json.dumps({"fixture-sv": {"type": "StatisticalVariable"}}))
    monkeypatch.setattr(m, "_FIXTURE_NODES_PATH", fixture_nodes_path)
    monkeypatch.setattr(
        m,
        "load_cases",
        lambda: [
            _case(
                "c1",
                {
                    "definite": {
                        "measure": {
                            "shape": {"measured_property": "amount"},
                            "stat_vars": ["expected-first", "shared"],
                        }
                    }
                },
            ),
            _case(
                "c2",
                {
                    "parts": [
                        {
                            "definite": {
                                "measure": {
                                    "shape": {"measured_property": "amount"},
                                    "stat_vars": ["shared", "expected-second"],
                                }
                            }
                        },
                        {"no_data": {"reason": "no_observations"}},
                    ]
                },
            ),
        ],
    )

    assert seed_sv_dcids() == ["expected-first", "expected-second", "fixture-sv", "shared"]


def test_seed_sv_dcids_returns_a_sorted_deduplicated_nonempty_list():
    seeds = seed_sv_dcids()
    assert seeds
    assert seeds == sorted(set(seeds))


# ---------------------------------------------------------------------------
# survey -- stub graph, no network
# ---------------------------------------------------------------------------


class _StubSurveyGraph:
    def __init__(self, arcs_by_dcid: dict[str, dict | None]):
        self._arcs_by_dcid = arcs_by_dcid
        self.batch_calls: list[list[str]] = []

    def node_arcs_batch(self, dcids: list[str]) -> dict[str, dict | None]:
        self.batch_calls.append(list(dcids))
        return {dcid: self._arcs_by_dcid.get(dcid) for dcid in dcids}


def test_survey_aggregates_resolved_unresolved_absent_across_svs():
    graph = _StubSurveyGraph(
        {
            "sv1": {
                "measurementDenominator": {"nodes": [{"dcid": "Count_Person"}]},
                "measurementQualifier": {"nodes": []},
                "constraintProperties": {"nodes": [{"dcid": "activitySource"}]},
                "activitySource": {"nodes": [{"name": "unresolved"}]},
            },
            "sv2": {
                "measurementDenominator": {"nodes": [{"name": "unresolved"}]},
                "measurementQualifier": {"nodes": []},
                "constraintProperties": {"nodes": [{"dcid": "activitySource"}]},
                "activitySource": {"nodes": [{"dcid": "Government"}]},
            },
            "sv3": None,  # unfetchable node: not classified, contributes nothing
        }
    )

    result = survey(graph, ["sv1", "sv2", "sv3"])

    assert result.counts["measurementDenominator"]["resolved"] == 1
    assert result.counts["measurementDenominator"]["unresolved"] == 1
    assert result.counts["measurementQualifier"]["absent"] == 2
    assert result.counts["activitySource"]["unresolved"] == 1
    assert result.counts["activitySource"]["resolved"] == 1
    # sv3's payload came back None, so it is not counted as classified even
    # though it was requested.
    assert result.n_classified == 2
    assert result.unresolved_pairs == [
        ("sv1", "activitySource"),
        ("sv2", "measurementDenominator"),
    ]


def test_survey_batches_and_throttles_between_batches(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(m.time, "sleep", lambda s: sleep_calls.append(s))

    dcids = [f"sv{i}" for i in range(m._BATCH_SIZE * 2 + 5)]
    graph = _StubSurveyGraph(dict.fromkeys(dcids))  # every SV absent; only batching matters here

    survey(graph, dcids)

    assert len(graph.batch_calls) == 3
    assert [len(b) for b in graph.batch_calls] == [m._BATCH_SIZE, m._BATCH_SIZE, 5]
    # Throttled between batches, not after the last one.
    assert sleep_calls == [m._THROTTLE_S, m._THROTTLE_S]


# ---------------------------------------------------------------------------
# reverse_probe -- stub graph, no network
# ---------------------------------------------------------------------------


class _StubReverseGraphTransport:
    """Stands in for `dc_graph.GraphClient`, the object `LiveGraphClient._client`
    holds and `scripts/survey_unresolved_arcs.py` posts through directly
    (`graph._client.post(...)`)."""

    def __init__(self, outer: _StubReverseGraph) -> None:
        self._outer = outer

    def post(
        self, path: str, payload: dict[str, Any], *, deadline: float | None = None
    ) -> dict[str, Any]:
        return self._outer._post(path, payload)


class _StubReverseGraph:
    _v2_base = "https://graph.example/core/api/v2"

    def __init__(
        self,
        pages: dict[object, dict[str, Any]],
        reverse_arcs: dict[tuple[str, str], dict[str, Any] | None] | None = None,
        reverse_pages: dict[tuple[str, str | None], dict[str, Any]] | None = None,
    ):
        self._pages = pages
        self._reverse_arcs = reverse_arcs or {}
        self._reverse_pages = reverse_pages or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.node_data_calls = 0
        self._client = _StubReverseGraphTransport(self)

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((url, payload))
        if payload["property"] == "<-typeOf":
            return self._pages[payload.get("nextToken")]

        paginated_body = self._reverse_pages.get((payload["property"], payload.get("nextToken")))
        if paginated_body is not None:
            return paginated_body

        data = {}
        for dcid in payload["nodes"]:
            arcs = self._reverse_arcs.get((dcid, payload["property"]))
            if arcs is not None:
                data[dcid] = {"arcs": arcs}
        return {"data": data}

    def _node_data(self, dcid: str, prop: str) -> dict | None:
        self.node_data_calls += 1
        return self._reverse_arcs.get((dcid, prop))


def test_reverse_probe_reports_no_instances_when_typeof_absent():
    graph = _StubReverseGraph({None: {"data": {}}})
    instance_dcids, n_hits = reverse_probe(graph, max_pages=1)
    assert instance_dcids == []
    assert n_hits == 0


def test_reverse_probe_counts_reverse_hits_across_instances(monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    graph = _StubReverseGraph(
        {
            None: {
                "data": {
                    "ProvisionalNode": {
                        "arcs": {
                            "typeOf": {"nodes": [{"dcid": "mortalityRate"}, {"dcid": "PerCapita"}]}
                        }
                    }
                }
            },
        },
        {
            ("mortalityRate", "<-measurementDenominator"): {
                "measurementDenominator": {"nodes": [{"dcid": "sv1"}]}
            },
            ("mortalityRate", "<-measurementQualifier"): None,
            ("PerCapita", "<-measurementDenominator"): None,
            ("PerCapita", "<-measurementQualifier"): None,
        },
    )

    instance_dcids, n_hits = reverse_probe(graph, max_pages=1)

    assert instance_dcids == ["mortalityRate", "PerCapita"]
    assert n_hits == 1
    assert graph.node_data_calls == 0


def test_reverse_probe_batches_reverse_properties_and_throttles_between_chunks(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(m.time, "sleep", lambda s: sleep_calls.append(s))
    instance_dcids = [f"provisional-{i}" for i in range(m._BATCH_SIZE * 2 + 1)]
    graph = _StubReverseGraph(
        {
            None: {
                "data": {
                    "ProvisionalNode": {
                        "arcs": {"typeOf": {"nodes": [{"dcid": dcid} for dcid in instance_dcids]}}
                    }
                }
            }
        },
        {
            ("provisional-0", "<-measurementDenominator"): {
                "measurementDenominator": {"nodes": [{"dcid": "sv-1"}, {"dcid": "sv-2"}]}
            },
            ("provisional-40", "<-measurementQualifier"): {
                "measurementQualifier": {"nodes": [{"dcid": "sv-3"}]}
            },
            ("provisional-80", "<-measurementDenominator"): {
                "measurementDenominator": {"nodes": [{"dcid": "sv-4"}]}
            },
        },
    )

    returned_dcids, n_hits = reverse_probe(graph, max_pages=1)

    reverse_calls = [call for call in graph.calls if call[1]["property"] != "<-typeOf"]
    assert returned_dcids == instance_dcids
    assert n_hits == 4
    assert graph.node_data_calls == 0
    assert len(reverse_calls) == 6
    assert [call[1]["property"] for call in reverse_calls] == [
        "<-measurementDenominator",
        "<-measurementQualifier",
    ] * 3
    assert [len(call[1]["nodes"]) for call in reverse_calls] == [
        m._BATCH_SIZE,
        m._BATCH_SIZE,
        m._BATCH_SIZE,
        m._BATCH_SIZE,
        1,
        1,
    ]
    assert sleep_calls == [m._THROTTLE_S, m._THROTTLE_S]


def test_reverse_probe_counts_all_reverse_property_pages(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(m.time, "sleep", lambda s: sleep_calls.append(s))
    graph = _StubReverseGraph(
        {None: {"data": {"ProvisionalNode": {"arcs": {"typeOf": {"nodes": [{"dcid": "first"}]}}}}}},
        reverse_pages={
            ("<-measurementDenominator", None): {
                "data": {
                    "first": {"arcs": {"measurementDenominator": {"nodes": [{"dcid": "sv-1"}]}}}
                },
                "nextToken": "denominator-page-2",
            },
            ("<-measurementDenominator", "denominator-page-2"): {
                "data": {
                    "first": {"arcs": {"measurementDenominator": {"nodes": [{"dcid": "sv-2"}]}}}
                }
            },
        },
    )

    instance_dcids, n_hits = reverse_probe(graph, max_pages=2)

    assert instance_dcids == ["first"]
    assert n_hits == 2
    assert graph.node_data_calls == 0
    assert graph.calls[1:3] == [
        (
            "/v2/node",
            {"nodes": ["first"], "property": "<-measurementDenominator"},
        ),
        (
            "/v2/node",
            {
                "nodes": ["first"],
                "property": "<-measurementDenominator",
                "nextToken": "denominator-page-2",
            },
        ),
    ]
    assert sleep_calls == [m._THROTTLE_S]


@pytest.mark.parametrize(
    ("max_pages", "pages", "message"),
    [
        (
            3,
            {
                ("<-measurementDenominator", None): {"data": {}, "nextToken": "same-token"},
                ("<-measurementDenominator", "same-token"): {
                    "data": {},
                    "nextToken": "same-token",
                },
            },
            "repeated",
        ),
        (
            1,
            {("<-measurementDenominator", None): {"data": {}, "nextToken": "second-page"}},
            "partial",
        ),
        (
            3,
            {("<-measurementDenominator", None): {"data": {}, "nextToken": 1}},
            "non-string",
        ),
    ],
)
def test_reverse_probe_rejects_invalid_reverse_property_pagination(
    monkeypatch,
    max_pages,
    pages,
    message,
):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    graph = _StubReverseGraph(
        {None: {"data": {"ProvisionalNode": {"arcs": {"typeOf": {"nodes": [{"dcid": "first"}]}}}}}},
        reverse_pages=pages,
    )

    with pytest.raises(m.ReversePaginationError, match=message):
        reverse_probe(graph, max_pages=max_pages)


def test_reverse_probe_stops_before_exhausting_shared_request_budget(monkeypatch):
    sleep_calls: list[float] = []
    monkeypatch.setattr(m.time, "sleep", lambda s: sleep_calls.append(s))
    instance_dcids = [f"provisional-{i}" for i in range(m._BATCH_SIZE + 1)]
    graph = _StubReverseGraph(
        {
            None: {
                "data": {
                    "ProvisionalNode": {
                        "arcs": {"typeOf": {"nodes": [{"dcid": dcid} for dcid in instance_dcids]}}
                    }
                }
            }
        },
        reverse_pages={
            ("<-measurementDenominator", None): {
                "data": {},
                "nextToken": "first-denominator-page-2",
            },
            ("<-measurementDenominator", "first-denominator-page-2"): {"data": {}},
        },
    )

    with pytest.raises(m.ReversePaginationError, match="request budget of 5 exhausted"):
        reverse_probe(graph, max_pages=2, max_requests=5)

    assert graph.node_data_calls == 0
    assert [call[1]["property"] for call in graph.calls] == [
        "<-typeOf",
        "<-measurementDenominator",
        "<-measurementDenominator",
        "<-measurementQualifier",
        "<-measurementDenominator",
    ]
    assert len(graph.calls) == 5
    assert sleep_calls == [m._THROTTLE_S, m._THROTTLE_S]


def test_reverse_probe_follows_top_level_next_token_in_first_seen_order(monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    graph = _StubReverseGraph(
        {
            None: {
                "data": {
                    "ProvisionalNode": {
                        "arcs": {"typeOf": {"nodes": [{"dcid": "first"}, {"dcid": "shared"}]}}
                    }
                },
                "nextToken": "second-page",
            },
            "second-page": {
                "data": {
                    "ProvisionalNode": {
                        "arcs": {"typeOf": {"nodes": [{"dcid": "shared"}, {"dcid": "last"}]}}
                    }
                }
            },
        }
    )
    instance_dcids, n_hits = reverse_probe(graph, max_pages=2)

    assert instance_dcids == ["first", "shared", "last"]
    assert n_hits == 0
    assert graph.calls[:2] == [
        (
            "/v2/node",
            {"nodes": ["ProvisionalNode"], "property": "<-typeOf"},
        ),
        (
            "/v2/node",
            {
                "nodes": ["ProvisionalNode"],
                "property": "<-typeOf",
                "nextToken": "second-page",
            },
        ),
    ]


def test_reverse_probe_refuses_to_report_a_page_limited_partial_enumeration(monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    graph = _StubReverseGraph(
        {
            None: {
                "data": {"ProvisionalNode": {"arcs": {"typeOf": {"nodes": [{"dcid": "first"}]}}}},
                "nextToken": "second-page",
            }
        }
    )

    with pytest.raises(m.ReversePaginationError, match="partial"):
        reverse_probe(graph, max_pages=1)


def test_reverse_probe_rejects_repeated_next_token_without_looping(monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    graph = _StubReverseGraph(
        {
            None: {
                "data": {"ProvisionalNode": {"arcs": {"typeOf": {"nodes": [{"dcid": "first"}]}}}},
                "nextToken": "same-token",
            },
            "same-token": {
                "data": {"ProvisionalNode": {"arcs": {"typeOf": {"nodes": [{"dcid": "second"}]}}}},
                "nextToken": "same-token",
            },
        }
    )

    with pytest.raises(m.ReversePaginationError, match="repeated"):
        reverse_probe(graph, max_pages=3)

    assert len(graph.calls) == 2


def test_positive_page_count_rejects_zero_negative_and_non_numeric_values():
    assert m._positive_page_count("1") == 1
    for value in ("0", "-1", "not-a-number"):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            m._positive_page_count(value)


def test_positive_request_count_rejects_zero_negative_and_non_numeric_values():
    assert m._positive_request_count("1") == 1
    for value in ("0", "-1", "not-a-number"):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            m._positive_request_count(value)


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_includes_survey_size_and_every_property():
    result = m.SurveyResult(
        counts={"measurementDenominator": Counter({"resolved": 2, "unresolved": 1, "absent": 0})},
        unresolved_pairs=[("sv2", "measurementDenominator")],
        n_classified=3,
    )
    report = format_report(result)
    assert "3 SVs classified" in report
    assert "measurementDenominator" in report


def test_format_report_lists_unresolved_sv_property_pairs():
    result = m.SurveyResult(
        counts={"measurementDenominator": Counter({"resolved": 1, "unresolved": 1, "absent": 0})},
        unresolved_pairs=[("sv2", "measurementDenominator")],
        n_classified=2,
    )
    report = format_report(result)
    assert "sv2  measurementDenominator" in report


def test_format_report_omits_pairs_section_when_none_are_unresolved():
    result = m.SurveyResult(
        counts={"measurementDenominator": Counter({"resolved": 2, "unresolved": 0, "absent": 0})},
        unresolved_pairs=[],
        n_classified=2,
    )
    report = format_report(result)
    assert "unresolved (sv_dcid, property) pairs" not in report
