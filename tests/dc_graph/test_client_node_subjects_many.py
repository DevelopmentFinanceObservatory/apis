"""Tests batched incoming-arc pagination, capping, and failure handling."""

import pytest

import dc_graph.client as client_module
from dc_graph import ArcSubject, GraphClient


def _subject_dcids(client: GraphClient, dcids: list[str], **kwargs) -> dict:
    """`node_subjects_many`'s answer."""
    return client.node_subjects_many(dcids, **kwargs)


def _arc_subject_dcids(client: GraphClient, dcids: list[str], **kwargs) -> dict:
    """`node_arc_subjects_many`'s answer with each subject reduced to its dcid."""
    return {
        lead: (tuple(subject.dcid for subject in subjects), flag)
        for lead, (subjects, flag) in client.node_arc_subjects_many(dcids, **kwargs).items()
    }


# Both batched reverse reads walk the same pages, so walk-shape tests run against each.
_BOTH_READS = pytest.mark.parametrize("read", [_subject_dcids, _arc_subject_dcids])


def test_node_subjects_many_request_shape(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        assert payload == {
            "nodes": ["CRSPurposeCode120", "CRSPurposeCode121"],
            "property": "<-specializationOf",
        }

        return {
            "data": {
                "CRSPurposeCode120": {
                    "arcs": {"specializationOf": {"nodes": [{"dcid": "CRSPurposeCode12010"}]}}
                },
                "CRSPurposeCode121": {
                    "arcs": {"specializationOf": {"nodes": [{"dcid": "CRSPurposeCode12110"}]}}
                },
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(
        ["CRSPurposeCode120", "CRSPurposeCode121"], property="specializationOf", max_results=25
    )
    assert result == {
        "CRSPurposeCode120": (("CRSPurposeCode12010",), "complete"),
        "CRSPurposeCode121": (("CRSPurposeCode12110",), "complete"),
    }


@_BOTH_READS
def test_node_subjects_many_absent_and_empty_dcids_map_to_empty_page(monkeypatch, read) -> None:
    """Absent dcids and nodes without the incoming arc return empty complete pages."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "leaf/NODE": {"arcs": {}},
                # "absent/NODE" omitted entirely from the response.
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = read(client, ["leaf/NODE", "absent/NODE"], property="typeOf", max_results=10)
    assert result == {
        "leaf/NODE": ((), "complete"),
        "absent/NODE": ((), "complete"),
    }


def test_node_subjects_many_dedups_a_subject_declared_by_two_provenances(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "Enum": {
                    "arcs": {
                        "typeOf": {
                            "nodes": [
                                {"dcid": "MemberA"},
                                {"dcid": "MemberA"},
                                {"dcid": "MemberB"},
                            ]
                        }
                    }
                }
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(["Enum"], property="typeOf", max_results=10)
    assert result == {"Enum": (("MemberA", "MemberB"), "complete")}


def test_node_subjects_many_multi_page_response_stitched(monkeypatch) -> None:
    """Pages concatenate in order and deduplicate subjects across the page boundary. The detail
    read collects a subject's provenances from both pages, sorted, and its distinct names."""
    calls: list[dict] = []
    oda = {
        "dcid": "DAC1AidType1010",
        "name": "official development assistance (DAC 1 aid type)",
        "types": ["DAC1AidTypeEnum"],
    }

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        if "nextToken" not in payload:
            return {
                "data": {
                    "DAC1AidTypeEnum": {
                        "arcs": {
                            "typeOf": {"nodes": [oda | {"provenanceId": "provenance/DFOOntology"}]}
                        }
                    },
                },
                "nextToken": "tok1",
            }
        assert payload["nextToken"] == "tok1"
        return {
            "data": {
                "DAC1AidTypeEnum": {
                    "arcs": {
                        "typeOf": {
                            "nodes": [
                                {"dcid": "DAC1AidType1010", "provenanceId": "dc/base/Schema"},
                                {"dcid": "DAC1AidType1015"},
                            ],
                        }
                    }
                }
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(["DAC1AidTypeEnum"], property="typeOf", max_results=10)
    detail = client.node_arc_subjects_many(["DAC1AidTypeEnum"], property="typeOf", max_results=10)
    assert result == {"DAC1AidTypeEnum": (("DAC1AidType1010", "DAC1AidType1015"), "complete")}
    assert detail == {
        "DAC1AidTypeEnum": (
            (
                ArcSubject(
                    dcid="DAC1AidType1010",
                    names=("official development assistance (DAC 1 aid type)",),
                    provenances=("dc/base/Schema", "provenance/DFOOntology"),
                ),
                ArcSubject(dcid="DAC1AidType1015", names=(), provenances=()),
            ),
            "complete",
        )
    }
    assert len(calls) == 4


def test_node_subjects_many_caps_at_max_results_and_stops_that_lead(monkeypatch) -> None:
    """A `(max_results + 1)`th subject marks the result capped at `max_results`."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "measuredValue": {
                    "arcs": {
                        "statType": {
                            "nodes": [{"dcid": f"sv/{i}"} for i in range(5)],
                        }
                    }
                }
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(["measuredValue"], property="statType", max_results=3)
    dcids, coverage = result["measuredValue"]
    assert coverage == "capped"
    assert dcids == ("sv/0", "sv/1", "sv/2")


def test_node_subjects_many_batch_pages_until_every_lead_caps_or_exhausts(monkeypatch) -> None:
    """Paging continues for uncapped leads and excludes leads that already reached their cap."""
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        if len(calls) == 1:
            return {
                "data": {
                    "fast/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "a"}, {"dcid": "b"}]}}},
                    "slow/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "x"}]}}},
                },
                "nextToken": "tok1",
            }
        # fast/LEAD capped (max_results=1) after page one and is no
        # longer named in the request; the data below stands in for
        # what a further page would carry for it, which this method
        # must never accumulate.
        return {
            "data": {
                "fast/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "c"}, {"dcid": "d"}]}}},
                "slow/LEAD": {"arcs": {"typeOf": {"nodes": []}}},
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(["fast/LEAD", "slow/LEAD"], property="typeOf", max_results=1)
    assert result["fast/LEAD"] == (("a",), "capped")
    assert result["slow/LEAD"] == (("x",), "complete")
    assert len(calls) == 2
    # The first request names the full batch; the second drops fast/LEAD,
    # which capped on page one, and names only slow/LEAD.
    assert calls[0]["nodes"] == ["fast/LEAD", "slow/LEAD"]
    assert calls[1]["nodes"] == ["slow/LEAD"]


def test_node_subjects_many_stops_paging_once_every_lead_has_capped(monkeypatch) -> None:
    """Paging stops when all leads cap, even if the mixer returns another token."""
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        return {
            "data": {
                "a/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "1"}, {"dcid": "2"}]}}},
                "b/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "3"}, {"dcid": "4"}]}}},
            },
            "nextToken": "tok-that-would-repeat-forever",
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(["a/LEAD", "b/LEAD"], property="typeOf", max_results=1)
    assert result["a/LEAD"] == (("1",), "capped")
    assert result["b/LEAD"] == (("3",), "capped")
    assert len(calls) == 1, "paging must stop as soon as every lead has capped"


def test_node_subjects_many_repeated_token_raises(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {"leaf/NODE": {"arcs": {"typeOf": {"nodes": []}}}},
            "nextToken": "tok1",
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    with pytest.raises(client_module.GraphReadError):
        client.node_subjects_many(["leaf/NODE"], property="typeOf", max_results=10)


def test_node_subjects_many_repeated_token_raises_the_substituted_class(monkeypatch) -> None:
    """Repeated tokens raise the configured `read_error_cls`."""

    class _CustomReadError(client_module.GraphReadError):
        pass

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {"leaf/NODE": {"arcs": {"typeOf": {"nodes": []}}}},
            "nextToken": "tok1",
        }

    client = GraphClient(base_url="http://fake/core/api", read_error_cls=_CustomReadError)
    monkeypatch.setattr(GraphClient, "post", fake_post)
    with pytest.raises(_CustomReadError):
        client.node_subjects_many(["leaf/NODE"], property="typeOf", max_results=10)


def test_node_subjects_many_page_budget_exhausted_reports_incomplete(monkeypatch) -> None:
    """A remaining token at the page bound reports `budget_exhausted` with a lower-bound count."""
    monkeypatch.setattr(client_module, "_NODE_SUBJECTS_MANY_MAX_PAGES", 2)
    responses = [
        {"data": {"leaf/NODE": {"arcs": {"typeOf": {"nodes": []}}}}, "nextToken": "tok1"},
        {"data": {"leaf/NODE": {"arcs": {"typeOf": {"nodes": []}}}}, "nextToken": "tok2"},
    ]
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        return responses[len(calls) - 1]

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(["leaf/NODE"], property="typeOf", max_results=10)
    assert result == {"leaf/NODE": ((), "budget_exhausted")}
    assert len(calls) == 2


def test_node_subjects_many_page_budget_exhausted_leaves_capped_lead_capped(monkeypatch) -> None:
    """At the page bound, a capped lead stays capped and an open lead reports `budget_exhausted`."""
    monkeypatch.setattr(client_module, "_NODE_SUBJECTS_MANY_MAX_PAGES", 2)
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        if len(calls) == 1:
            return {
                "data": {
                    "capped/LEAD": {
                        "arcs": {
                            "typeOf": {
                                "nodes": [
                                    {"dcid": "a"},
                                    {"dcid": "b"},
                                    {"dcid": "c"},
                                ]
                            }
                        }
                    },
                    "open/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "x"}]}}},
                },
                "nextToken": "tok1",
            }
        # capped/LEAD dropped from the request after page one; only
        # open/LEAD is named, and it still has more to give.
        assert payload["nodes"] == ["open/LEAD"]
        return {
            "data": {"open/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "y"}]}}}},
            "nextToken": "tok2",
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(
        ["capped/LEAD", "open/LEAD"], property="typeOf", max_results=2
    )
    assert result["capped/LEAD"] == (("a", "b"), "capped")
    assert result["open/LEAD"] == (("x", "y"), "budget_exhausted")
    assert len(calls) == 2


@_BOTH_READS
def test_node_subjects_many_drops_capped_lead_from_later_pages_to_drain(monkeypatch, read) -> None:
    """Capped leads leave later requests, so their paging cannot exhaust the shared page budget."""
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        nodes = payload["nodes"]
        if len(calls) > 1 and "capped/LEAD" in nodes:
            raise AssertionError(
                "capped/LEAD must not remain in the request body past the page where it capped"
            )

        if len(calls) == 1:
            return {
                "data": {
                    "capped/LEAD": {
                        "arcs": {"typeOf": {"nodes": [{"dcid": "c1"}, {"dcid": "c2"}]}}
                    },
                    "wide/LEAD": {"arcs": {"typeOf": {"nodes": [{"dcid": "w1"}]}}},
                },
                "nextToken": "tok1",
            }
        # capped/LEAD is gone from the request; wide/LEAD's own
        # exhaustion (no further nodes, no nextToken) is all that
        # remains to end paging.
        return {"data": {"wide/LEAD": {"arcs": {"typeOf": {"nodes": []}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = read(client, ["capped/LEAD", "wide/LEAD"], property="typeOf", max_results=1)
    assert result["capped/LEAD"] == (("c1",), "capped")
    assert result["wide/LEAD"] == (("w1",), "complete")
    assert len(calls) == 2
    assert calls[0]["nodes"] == ["capped/LEAD", "wide/LEAD"]
    assert calls[1]["nodes"] == ["wide/LEAD"]


def test_node_subjects_many_empty_dcids_returns_empty_dict_without_network_call(
    monkeypatch,
) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        raise AssertionError("must not POST for an empty dcids list")

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    assert client.node_subjects_many([], property="typeOf", max_results=10) == {}


def test_node_subjects_many_deduplicates_repeated_input_dcids(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        return {"data": {"leaf/NODE": {"arcs": {"typeOf": {"nodes": []}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_subjects_many(
        ["leaf/NODE", "leaf/NODE"], property="typeOf", max_results=10
    )
    assert result == {"leaf/NODE": ((), "complete")}
    assert calls[0]["nodes"] == ["leaf/NODE"]


def test_node_subjects_many_rejects_unsafe_property_without_a_network_call(monkeypatch) -> None:
    """A property carrying a character the <-{property} inline syntax
    cannot safely embed raises ValueError before any POST is issued."""

    def fake_post(self, path, payload, *, deadline=None):
        raise AssertionError("must not POST for an unsafe property")

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    with pytest.raises(ValueError, match="typeOf"):
        client.node_subjects_many(["leaf/NODE"], property="typeOf{x:y}", max_results=10)


_TWO_NAMES = [
    {"dcid": "crsPurpose", "name": "CRS purpose", "provenanceId": "provenance/DFOOntology"},
    {"dcid": "crsPurpose", "name": "Purpose", "provenanceId": "dc/base/Schema"},
]


@pytest.mark.parametrize("entries", [_TWO_NAMES, _TWO_NAMES[::-1]], ids=["dfo-first", "base-first"])
@pytest.mark.parametrize("batched", [True, False], ids=["many", "single"])
def test_arc_subjects_carry_every_name_whatever_the_wire_order(
    monkeypatch, entries: list[dict], batched: bool
) -> None:
    """Two provenances name one subject differently; the subject carries both names, sorted."""

    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {"Property": {"arcs": {"typeOf": {"nodes": entries}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    if batched:
        ((subjects, _flag),) = client.node_arc_subjects_many(
            ["Property"], property="typeOf", max_results=10
        ).values()
    else:
        subjects = client.node_arc_subjects("Property", property="typeOf", max_results=10).subjects

    assert [(subject.names, subject.provenances) for subject in subjects] == [
        (("CRS purpose", "Purpose"), ("dc/base/Schema", "provenance/DFOOntology"))
    ]


def test_plain_subject_reads_hand_no_entry_to_any_fold(monkeypatch) -> None:
    """`node_subjects` and `node_subjects_many` walk with no fold, so the walk keeps only
    dcids; the arc variants walk the same pages with one."""
    folds: list[tuple[str, object]] = []
    walk = GraphClient._walk_subjects
    walk_many = GraphClient._walk_subjects_many

    def spy(self, *args, fold, **kwargs):
        folds.append(("single", fold))
        return walk(self, *args, fold=fold, **kwargs)

    def spy_many(self, *args, fold, **kwargs):
        folds.append(("many", fold))
        return walk_many(self, *args, fold=fold, **kwargs)

    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {"Property": {"arcs": {"typeOf": {"nodes": _TWO_NAMES}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    monkeypatch.setattr(GraphClient, "_walk_subjects", spy)
    monkeypatch.setattr(GraphClient, "_walk_subjects_many", spy_many)
    client = GraphClient(base_url="http://fake/core/api")

    plain = client.node_subjects("Property", property="typeOf", max_results=10)
    plain_many = client.node_subjects_many(["Property"], property="typeOf", max_results=10)
    client.node_arc_subjects("Property", property="typeOf", max_results=10)
    client.node_arc_subjects_many(["Property"], property="typeOf", max_results=10)

    assert plain.dcids == ("crsPurpose",)
    assert plain_many == {"Property": (("crsPurpose",), "complete")}
    assert [(walked, fold is None) for walked, fold in folds] == [
        ("single", True),
        ("many", True),
        ("single", False),
        ("many", False),
    ]
