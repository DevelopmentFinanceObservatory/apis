"""Unit tests for GraphClient's shared node_arcs / node_arcs_batch cache,
and node_observation_properties_batch's own cache.

Covers partial cache hits, the single-dcid confirmed-empty fast path versus
the multi-dcid retry, omission recovery and its persistent-omission error,
LRU eviction, and that node_arcs (single-dcid) and node_arcs_batch read and
write the exact same cache. Offline only -- the transport is monkeypatched.
"""

import pytest

from dc_graph import GraphClient, GraphReadError
from graph_generation.stamp import UNKNOWN_KEY


def test_node_arcs_batch_partial_hit_posts_only_misses(monkeypatch) -> None:
    """A cached dcid is served from the arcs cache; only misses appear in the POST."""
    posted_nodes: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        posted_nodes.append(list(payload.get("nodes", [])))
        return {
            "data": {
                "country/DEU": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )
    cached_arcs = {"typeOf": {"nodes": [{"dcid": "Country"}]}}
    client._arcs_cache[(UNKNOWN_KEY, "country/KEN")] = cached_arcs

    result = client.node_arcs_batch(["country/KEN", "country/DEU"])

    assert result["country/KEN"] is cached_arcs, "Cache hit must be returned as-is"
    assert result["country/DEU"] == {"typeOf": {"nodes": [{"dcid": "Country"}]}}
    assert len(posted_nodes) == 1
    assert posted_nodes[0] == ["country/DEU"], "POST must include only the miss"
    assert client._arcs_cache.get((UNKNOWN_KEY, "country/DEU")) is not None


def test_node_arcs_batch_confirmed_empty_dcid_trusted_and_cached(monkeypatch) -> None:
    """A lone present-but-empty dcid maps to `None` without a retry and caches as `None`."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        # The dcid's key is present -- distinguishes a confirmed empty node from
        # an omission -- but its arcs are empty.
        return {"data": {"empty/NODE": {"arcs": {}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result1 = client.node_arcs_batch(["empty/NODE"])
    result2 = client.node_arcs_batch(["empty/NODE"])

    assert result1 == {"empty/NODE": None}
    assert result2 == {"empty/NODE": None}
    assert call_count[0] == 1, "a lone dcid's empty response is trusted and the warm read is free"
    assert (UNKNOWN_KEY, "empty/NODE") in client._arcs_cache
    assert client._arcs_cache[(UNKNOWN_KEY, "empty/NODE")] is None


def test_node_arcs_batch_confirmed_empty_dcid_in_multi_batch_retries_then_caches(
    monkeypatch,
) -> None:
    """An empty response retries when the batch holds several dcids; the retry's
    confirmed-empty answer caches as `None`, so a warm read makes no request."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        calls.append(nodes)
        if len(calls) == 1:
            return {
                "data": {
                    "empty/NODE": {"arcs": {}},
                    "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                }
            }
        assert nodes == ["empty/NODE"]
        return {"data": {"empty/NODE": {"arcs": {}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_batch(["empty/NODE", "country/KEN"])

    assert result["empty/NODE"] is None
    assert len(calls) == 2, "the empty entry retries because the batch held more than one dcid"
    assert client._arcs_cache[(UNKNOWN_KEY, "empty/NODE")] is None

    warm = client.node_arcs_batch(["empty/NODE", "country/KEN"])

    assert warm["empty/NODE"] is None
    assert warm["country/KEN"] == result["country/KEN"]
    assert len(calls) == 2, "a warm read of confirmed-empty and positive entries makes no request"


def test_node_arcs_batch_group_retry_empty_is_returned_but_not_cached(monkeypatch) -> None:
    """Two suspects that come back empty together from the group retry map to `None` but stay
    uncached: the retry shared a batch, so the empty answer may depend on its composition."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        calls.append(nodes)
        return {"data": {dcid: {"arcs": {}} for dcid in nodes}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_batch(["empty/A", "empty/B"])

    assert result == {"empty/A": None, "empty/B": None}
    assert calls == [["empty/A", "empty/B"], ["empty/A", "empty/B"]]
    assert (UNKNOWN_KEY, "empty/A") not in client._arcs_cache
    assert (UNKNOWN_KEY, "empty/B") not in client._arcs_cache


def test_node_arcs_batch_omission_recovered_by_retry(monkeypatch) -> None:
    """A dcid omitted from a batch is recovered by a singleton retry and cached."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        calls.append(nodes)
        if len(calls) == 1:
            # Initial batch: "asia" answers; "europe" is absent from the response.
            assert set(nodes) == {"asia", "europe"}
            return {"data": {"asia": {"arcs": {"name": {"nodes": [{"value": "Asia"}]}}}}}
        # Retry: "europe" alone now answers with its real arcs.
        assert nodes == ["europe"]
        return {"data": {"europe": {"arcs": {"name": {"nodes": [{"value": "Europe"}]}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_batch(["asia", "europe"])

    assert result["asia"] == {"name": {"nodes": [{"value": "Asia"}]}}
    assert result["europe"] == {"name": {"nodes": [{"value": "Europe"}]}}
    assert len(calls) == 2
    assert client._arcs_cache[(UNKNOWN_KEY, "europe")] == {"name": {"nodes": [{"value": "Europe"}]}}


def test_node_arcs_batch_empty_arcs_entry_recovered_by_retry(monkeypatch) -> None:
    """A present-but-empty entry in a multi-dcid batch is retried and can recover real arcs."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        calls.append(nodes)
        if len(calls) == 1:
            return {
                "data": {
                    "europe": {"arcs": {}},
                    "asia": {"arcs": {"name": {"nodes": [{"value": "Asia"}]}}},
                }
            }
        assert nodes == ["europe"]
        return {"data": {"europe": {"arcs": {"name": {"nodes": [{"value": "Europe"}]}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_batch(["europe", "asia"])

    assert result["europe"] == {"name": {"nodes": [{"value": "Europe"}]}}
    assert len(calls) == 2
    assert client._arcs_cache[(UNKNOWN_KEY, "europe")] == {"name": {"nodes": [{"value": "Europe"}]}}


def test_node_arcs_batch_single_dcid_empty_response_trusted_without_retry(monkeypatch) -> None:
    """A batch that originally held exactly one dcid, present but with an
    empty arcs map, is trusted immediately -- there is no other dcid in the
    request for the empty result to depend on, so no retry fires."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return {"data": {"europe": {"arcs": {}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_batch(["europe"])

    assert result["europe"] is None
    assert len(calls) == 1, "a lone dcid's empty response must not trigger a retry"
    assert client._arcs_cache[(UNKNOWN_KEY, "europe")] is None


def test_node_arcs_batch_omission_persists_raises(monkeypatch) -> None:
    """A dcid still absent from the retry response is a protocol violation: it
    raises GraphReadError naming the dcid, and nothing is cached."""

    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {}}  # never answers, initial or retry

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError, match=r"never/ANSWERS"):
        client.node_arcs_batch(["never/ANSWERS"])

    assert (UNKNOWN_KEY, "never/ANSWERS") not in client._arcs_cache


def test_node_arcs_batch_omitted_dcid_is_re_read_not_cached(monkeypatch) -> None:
    """A dcid the response omits is never cached as empty: after the failed read, the
    next call POSTs it again and recovers its real arcs."""
    calls: list[list[str]] = []
    answering = [False]

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        if not answering[0]:
            return {"data": {}}
        return {"data": {"late/NODE": {"arcs": {"name": {"nodes": [{"value": "Late"}]}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError):
        client.node_arcs_batch(["late/NODE"])
    assert (UNKNOWN_KEY, "late/NODE") not in client._arcs_cache
    posts_after_failure = len(calls)

    answering[0] = True
    result = client.node_arcs_batch(["late/NODE"])

    assert result["late/NODE"] == {"name": {"nodes": [{"value": "Late"}]}}
    assert len(calls) == posts_after_failure + 1, "the omitted dcid was read again"


def test_node_arcs_batch_omission_never_cached_even_alongside_a_good_dcid(monkeypatch) -> None:
    """A batch mixing a confirmed dcid with one that never answers raises for the
    unanswered dcid; the confirmed dcid's real arcs are unaffected, and the cache
    holds only that confirmed entry regardless of what else was in the batch."""

    def fake_post(self, path, payload, *, deadline=None):
        nodes = payload["nodes"]
        return {
            "data": {
                dcid: {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}
                for dcid in nodes
                if dcid == "country/KEN"
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError):
        client.node_arcs_batch(["country/KEN", "never/ANSWERS"])

    assert client._arcs_cache.get((UNKNOWN_KEY, "country/KEN")) == {
        "typeOf": {"nodes": [{"dcid": "Country"}]}
    }
    assert (UNKNOWN_KEY, "never/ANSWERS") not in client._arcs_cache


def test_node_arcs_batch_error_before_cache_write(monkeypatch) -> None:
    """A GraphReadError from _post propagates without any cache write."""

    def fake_post(self, path, payload, *, deadline=None):
        raise GraphReadError("simulated 503", upstream_status=503)

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError):
        client.node_arcs_batch(["country/KEN"])

    assert (UNKNOWN_KEY, "country/KEN") not in client._arcs_cache


def test_node_arcs_batch_lru_eviction_at_maxsize(monkeypatch) -> None:
    """The least-recently-used entry is evicted once the cache exceeds maxsize."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                dcid: {"arcs": {"typeOf": {"nodes": [{"dcid": "Thing"}]}}}
                for dcid in payload.get("nodes", [])
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    # Pin the arcs cache's own maxsize below its default at construction.
    # cachetools.TLRUCache's constructor-time maxsize is the only bound the
    # cache enforces (see dc_graph.GraphClient's own comment beside _arcs_cache),
    # so shrinking it after construction is not possible.
    client = GraphClient(base_url="http://fake/core/api", arcs_cache_size=2)

    client.node_arcs_batch(["a/1"])
    client.node_arcs_batch(["a/2"])
    assert list(client._arcs_cache.keys()) == [(UNKNOWN_KEY, "a/1"), (UNKNOWN_KEY, "a/2")]

    client.node_arcs_batch(["a/3"])
    assert (UNKNOWN_KEY, "a/1") not in client._arcs_cache, (
        "Least-recently-used entry must be evicted"
    )
    assert list(client._arcs_cache.keys()) == [(UNKNOWN_KEY, "a/2"), (UNKNOWN_KEY, "a/3")]


def test_node_arcs_single_call_none_on_confirmed_empty(monkeypatch) -> None:
    """node_arcs (delegating to node_arcs_batch with a single dcid) returns
    None for a dcid present with an empty arcs map, trusted immediately
    without a retry, and caches it (a later call makes no POST)."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return {"data": {"empty/NODE": {"arcs": {}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    assert client.node_arcs("empty/NODE") is None
    assert client.node_arcs("empty/NODE") is None
    assert call_count[0] == 1


def test_node_arcs_single_call_raises_on_persistent_omission(monkeypatch) -> None:
    """node_arcs (delegating to node_arcs_batch) raises, rather than returning a
    false None, for a dcid still omitted from the retry response."""

    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError):
        client.node_arcs("absent/NODE")


def test_node_arcs_and_node_arcs_batch_share_one_cache(monkeypatch) -> None:
    """`node_arcs` and `node_arcs_batch` share one cache."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return {"data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    arcs_single = client.node_arcs("country/KEN")
    assert arcs_single is not None
    assert call_count[0] == 1

    result_batch = client.node_arcs_batch(["country/KEN"])
    assert result_batch["country/KEN"] is arcs_single
    assert call_count[0] == 1, "node_arcs_batch must hit the cache node_arcs populated"


def test_exists_batch_recovers_grouped_false_empties_with_singleton_evidence(monkeypatch) -> None:
    """Singleton `->*` probes confirm grouped misses; namespace-shaped ids receive no exemption."""
    calls: list[tuple[str, list[str]]] = []
    known = {
        "measuredValue",
        "country/ARE",
        "sdg/EG_FEC_RNEW",
        "worldBank/4_1_SHARE_RE_IN_ELECTRICITY",
        "nuts/FI2",
    }
    requested = [
        "measuredValue",
        "country/ARE",
        "sdg/EG_FEC_RNEW",
        "worldBank/4_1_SHARE_RE_IN_ELECTRICITY",
        "nuts/FI2",
        "country/ZZZ",
        "sdg/NOT_A_REAL_INDICATOR",
        "worldBank/NOT_A_REAL_INDICATOR",
        "nuts/NOT_REAL",
        "country/ARE",
    ]

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        property_name = payload["property"]
        calls.append((property_name, nodes))
        if property_name == "->*":
            assert len(nodes) == 1, "exists_batch must never group arc confirmation probes"
            dcid = nodes[0]
            arcs = {"name": {"nodes": [{"value": dcid}]}} if dcid in known else {}
            return {"data": {dcid: {"arcs": arcs}}}

        assert property_name == "->name"
        # ``measuredValue`` remains a normal low-payload batch positive. The
        # rest omit a name, just as a grouped ``->*`` false-empty would; they
        # must get authoritative singleton arc checks instead of an absence.
        return {
            "data": {
                dcid: {
                    "arcs": (
                        {"name": {"nodes": [{"value": dcid}]}} if dcid == "measuredValue" else {}
                    )
                }
                for dcid in nodes
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    assert client.exists_batch(requested) == known
    assert calls[0] == ("->name", list(dict.fromkeys(requested)))
    # exists_batch does not send a large ``->*`` batch: it uses its one
    # low-payload name batch, then exactly one singleton probe per apparent
    # miss. The singleton-confirmed fakes remain absent.
    assert calls[1:] == [("->*", [dcid]) for dcid in calls[0][1][1:]]


def test_exists_batch_does_not_probe_singletons_for_batch_positives(monkeypatch) -> None:
    """The normal positive path stays one deduplicated batch request."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        calls.append(nodes)
        return {"data": {dcid: {"arcs": {"name": {"nodes": [{"value": dcid}]}}} for dcid in nodes}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    assert client.exists_batch(["Country", "measuredValue", "Country"]) == {
        "Country",
        "measuredValue",
    }
    assert calls == [["Country", "measuredValue"]]


def test_exists_batch_propagates_singleton_confirmation_errors(monkeypatch) -> None:
    """An apparent miss is never silently called absent when its probe fails."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        calls.append(nodes)
        if len(nodes) == 1:
            raise GraphReadError("singleton probe failed", upstream_status=503)
        return {"data": {dcid: {"arcs": {}} for dcid in nodes}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError, match="singleton probe failed"):
        client.exists_batch(["country/ARE", "sdg/EG_FEC_RNEW"])

    assert calls == [["country/ARE", "sdg/EG_FEC_RNEW"], ["country/ARE"]]


@pytest.mark.parametrize(
    "malformed_node_data",
    [
        "not a mapping",
        ["not", "a", "mapping"],
        "",
        [],
        False,
        {"arcs": "not a mapping"},
        {"arcs": ["not", "a", "mapping"]},
        {"arcs": ""},
        {"arcs": []},
        {"arcs": False},
    ],
    ids=[
        "node-string",
        "node-list",
        "node-empty-string",
        "node-empty-list",
        "node-false",
        "arcs-string",
        "arcs-list",
        "arcs-empty-string",
        "arcs-empty-list",
        "arcs-false",
    ],
)
def test_exists_batch_rejects_malformed_singleton_arc_evidence(
    monkeypatch, malformed_node_data
) -> None:
    """Malformed singleton payloads are infrastructure failures, never absence."""
    calls: list[tuple[str, list[str]]] = []

    def fake_post(self, path, payload, *, deadline=None):
        property_name = payload["property"]
        nodes = list(payload["nodes"])
        calls.append((property_name, nodes))
        if property_name == "->name":
            # A well-formed no-label result reaches the singleton fallback.
            return {"data": {"country/ARE": {"arcs": {}}}}
        assert property_name == "->*"
        assert nodes == ["country/ARE"]
        return {"data": {"country/ARE": malformed_node_data}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError, match=r"malformed (node data|arcs)"):
        client.exists_batch(["country/ARE"])

    assert calls == [("->name", ["country/ARE"]), ("->*", ["country/ARE"])]
    assert client._arcs_cache == {}


# ---------------------------------------------------------------------------
# node_arcs_batch / node_arcs -- raw wire dedup (#698)
# ---------------------------------------------------------------------------


def test_node_arcs_batch_dedups_a_value_declared_by_two_provenances(monkeypatch) -> None:
    """Duplicates keep the first raw entry and its `provenanceId`."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/DUAL": {
                    "arcs": {
                        "isPartOf": {
                            "nodes": [
                                {"dcid": "Agg", "provenanceId": "dc/base/India_LifeExpectancy"},
                                {"dcid": "Agg", "provenanceId": "dc/base/Schema"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_batch(["sv/DUAL"])

    assert result["sv/DUAL"] == {
        "isPartOf": {"nodes": [{"dcid": "Agg", "provenanceId": "dc/base/India_LifeExpectancy"}]}
    }


def test_node_arcs_batch_dedup_lands_in_the_cache(monkeypatch) -> None:
    """The cache stores deduplicated arcs."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/DUAL": {
                    "arcs": {
                        "typeOf": {
                            "nodes": [
                                {"dcid": "StatisticalVariable"},
                                {"dcid": "StatisticalVariable"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    client.node_arcs_batch(["sv/DUAL"])

    assert client._arcs_cache[(UNKNOWN_KEY, "sv/DUAL")] == {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}
    }


def test_node_arcs_batch_preserves_one_entry_carrying_neither_dcid_nor_value(
    monkeypatch,
) -> None:
    """A keyless arc entry survives once as the raw presence signal for an unresolved target.

    A resolvable entry on the same property also survives in first-occurrence order.
    """

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/PARTIAL": {
                    "arcs": {
                        "measurementQualifier": {
                            "nodes": [
                                {"provenanceId": "dc/base/A"},
                                {"dcid": "Provisional", "provenanceId": "dc/base/B"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_batch(["sv/PARTIAL"])

    assert result["sv/PARTIAL"] == {
        "measurementQualifier": {
            "nodes": [
                {"provenanceId": "dc/base/A"},
                {"dcid": "Provisional", "provenanceId": "dc/base/B"},
            ]
        }
    }


def test_node_arcs_batch_dedups_multiple_entries_carrying_neither_field_to_one(
    monkeypatch,
) -> None:
    """Duplicate keyless entries collapse to one unresolved-node presence signal."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/UNRESOLVABLE": {
                    "arcs": {
                        "measurementQualifier": {
                            "nodes": [
                                {"provenanceId": "dc/base/A"},
                                {"provenanceId": "dc/base/B"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_batch(["sv/UNRESOLVABLE"])

    assert result["sv/UNRESOLVABLE"] == {
        "measurementQualifier": {"nodes": [{"provenanceId": "dc/base/A"}]}
    }


def test_node_arcs_dedups_via_the_shared_batch_path(monkeypatch) -> None:
    """node_arcs (single-dcid) inherits node_arcs_batch's dedup, since it
    delegates to node_arcs_batch for one dcid."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/DUAL": {
                    "arcs": {
                        "measuredProperty": {
                            "nodes": [{"dcid": "amount"}, {"dcid": "amount"}],
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs("sv/DUAL")

    assert result == {"measuredProperty": {"nodes": [{"dcid": "amount"}]}}


# ---------------------------------------------------------------------------
# node_arcs_many
# ---------------------------------------------------------------------------


def test_node_arcs_many_dedups_a_value_declared_by_two_provenances(monkeypatch) -> None:
    # (#698) Both readers collapse the same shape value declared by two provenances.
    def fake_post(self, path, payload, *, deadline=None):
        return {
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

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_arcs_many(["lifeExpectancy"], properties=["measuredProperty"])

    assert result == {"lifeExpectancy": {"measuredProperty": ("lifeExpectancy",)}}


def test_node_arcs_many_rejects_unsafe_property_without_a_network_call(monkeypatch) -> None:
    """Unsafe inline property syntax raises `ValueError` before POST."""

    def fake_post(self, path, payload, *, deadline=None):
        raise AssertionError("must not POST for an unsafe property")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(ValueError):
        client.node_arcs_many(["lifeExpectancy"], properties=["measuredProperty", "a]b"])


# ---------------------------------------------------------------------------
# node_observation_properties_batch
# ---------------------------------------------------------------------------


def test_node_observation_properties_batch_present_arc_returns_tuple_in_api_order(
    monkeypatch,
) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        assert payload["property"] == "->observationProperties"
        return {
            "data": {
                "sv/ROLE_BEARING": {
                    "arcs": {
                        "observationProperties": {
                            "nodes": [
                                {"dcid": "financingProvider"},
                                {"dcid": "financingRecipient"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_observation_properties_batch(["sv/ROLE_BEARING"])

    assert result == {"sv/ROLE_BEARING": ("financingProvider", "financingRecipient")}


def test_node_observation_properties_batch_absent_arc_and_nonexistent_node_return_empty_tuple(
    monkeypatch,
) -> None:
    """Nodes without `observationProperties` return `()` without retry."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return {
            "data": {
                "sv/STANDARD": {"arcs": {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}}},
                "does/not/exist": {"arcs": {}},
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_observation_properties_batch(["sv/STANDARD", "does/not/exist"])

    assert result == {"sv/STANDARD": (), "does/not/exist": ()}
    assert len(calls) == 1, "a present-but-empty entry must not trigger a retry"

    warm = client.node_observation_properties_batch(["sv/STANDARD", "does/not/exist"])

    assert warm == result
    assert len(calls) == 1, "the empty results are served from the cache"


def test_node_observation_properties_batch_dedup_and_completeness(monkeypatch) -> None:
    """Duplicate inputs collapse; every unique dcid gets a result."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return {
            "data": {
                "sv/A": {"arcs": {"observationProperties": {"nodes": [{"dcid": "role/X"}]}}},
                "sv/B": {"arcs": {}},
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_observation_properties_batch(["sv/A", "sv/B", "sv/A"])

    assert calls == [["sv/A", "sv/B"]], "duplicates must be deduplicated before the POST"
    assert result == {"sv/A": ("role/X",), "sv/B": ()}


def test_node_observation_properties_batch_dedups_a_target_declared_by_two_provenances(
    monkeypatch,
) -> None:
    # (#698) Duplicate role targets would make positional role-axis assignment collide.
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/ROLE_BEARING": {
                    "arcs": {
                        "observationProperties": {
                            "nodes": [
                                {"dcid": "financingProvider"},
                                {"dcid": "financingProvider"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_observation_properties_batch(["sv/ROLE_BEARING"])

    assert result == {"sv/ROLE_BEARING": ("financingProvider",)}


def test_node_observation_properties_batch_omission_recovered_by_retry(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        calls.append(nodes)
        if len(calls) == 1:
            assert set(nodes) == {"sv/A", "sv/MISSING"}
            return {"data": {"sv/A": {"arcs": {}}}}
        assert nodes == ["sv/MISSING"]
        return {
            "data": {
                "sv/MISSING": {"arcs": {"observationProperties": {"nodes": [{"dcid": "role/Y"}]}}}
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.node_observation_properties_batch(["sv/A", "sv/MISSING"])

    assert result == {"sv/A": (), "sv/MISSING": ("role/Y",)}
    assert len(calls) == 2


def test_node_observation_properties_batch_omission_persists_raises(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {}}  # never answers, initial or retry

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError, match=r"never/ANSWERS"):
        client.node_observation_properties_batch(["never/ANSWERS"])

    assert (UNKNOWN_KEY, "never/ANSWERS") not in client._observation_properties_flight._store


def test_node_observation_properties_batch_cache_hit_skips_post(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return {
            "data": {"sv/B": {"arcs": {"observationProperties": {"nodes": [{"dcid": "role/Z"}]}}}}
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )
    client._observation_properties_flight._store[(UNKNOWN_KEY, "sv/A")] = (
        float("inf"),
        ("role/X",),
    )

    result = client.node_observation_properties_batch(["sv/A", "sv/B"])

    assert result == {"sv/A": ("role/X",), "sv/B": ("role/Z",)}
    assert calls == [["sv/B"]], "the cached dcid must not appear in the POST"


@pytest.mark.parametrize(
    "malformed_body",
    [
        "not a mapping",
        {"data": "not a mapping"},
        {"data": {"sv/A": "not a mapping"}},
        {"data": {"sv/A": {"arcs": "not a mapping"}}},
    ],
    ids=["body-not-a-mapping", "data-not-a-mapping", "node-not-a-mapping", "arcs-not-a-mapping"],
)
def test_node_observation_properties_batch_malformed_body_raises(
    monkeypatch, malformed_body
) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return malformed_body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    with pytest.raises(GraphReadError):
        client.node_observation_properties_batch(["sv/A"])


def test_resolve_by_arc_maps_every_name_preserves_order_empty_list_for_no_candidates(
    monkeypatch,
) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "node": "Kenya",
                    "candidates": [{"dcid": "country/KEN"}, {"dcid": "country/KEN2"}],
                },
                {"node": "Nowhereland", "candidates": []},
            ]
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.resolve_by_arc(["Kenya", "Nowhereland"], arc="<-name->dcid")

    assert result == {"Kenya": ["country/KEN", "country/KEN2"], "Nowhereland": []}


def test_resolve_by_arc_omitted_name_maps_to_empty_list_unrequested_node_ignored(
    monkeypatch,
) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {"node": "Togo", "candidates": [{"dcid": "country/TGO"}]},
                {"node": "Unrequested", "candidates": [{"dcid": "country/ZZZ"}]},
            ]
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.resolve_by_arc(["Togo", "Kenya"], arc="<-name->dcid")

    assert result == {"Togo": ["country/TGO"], "Kenya": []}


def test_resolve_by_arc_candidate_without_dcid_is_skipped(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "node": "Kenya",
                    "candidates": [{"dcid": "country/KEN"}, {"dominantType": "Country"}],
                }
            ]
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.resolve_by_arc(["Kenya"], arc="<-name->dcid")

    assert result == {"Kenya": ["country/KEN"]}


def test_resolve_by_arc_dedups_a_candidate_declared_by_two_provenances(monkeypatch) -> None:
    # (#698) A candidate declared by two Data Commons provenances reports
    # the same dcid twice among a name's candidates.
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "node": "Kenya",
                    "candidates": [{"dcid": "country/KEN"}, {"dcid": "country/KEN"}],
                }
            ]
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.resolve_by_arc(["Kenya"], arc="<-name->dcid")

    assert result == {"Kenya": ["country/KEN"]}


def test_resolve_by_arc_keeps_genuinely_distinct_candidates(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "node": "Congo",
                    "candidates": [{"dcid": "country/COD"}, {"dcid": "country/COG"}],
                }
            ]
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    result = client.resolve_by_arc(["Congo"], arc="<-name->dcid")

    assert result == {"Congo": ["country/COD", "country/COG"]}


def test_resolve_by_arc_empty_names_returns_empty_dict_no_request(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload.get("nodes", [])))
        return {"entities": []}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    assert client.resolve_by_arc([], arc="<-name->dcid") == {}
    assert calls == []


def test_resolve_by_arc_request_shape_single_post(monkeypatch) -> None:
    posted: list[tuple[str, dict]] = []

    def fake_post(self, path, payload, *, deadline=None):
        posted.append((path, payload))
        return {"entities": []}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", arcs_cache_size=10000, obs_properties_cache_size=10000
    )

    client.resolve_by_arc(["Kenya", "Togo"], arc="<-name->dcid")

    assert len(posted) == 1
    path, payload = posted[0]
    assert path == "/v2/resolve"
    assert payload == {"nodes": ["Kenya", "Togo"], "property": "<-name->dcid"}


# ---------------------------------------------------------------------------
# Import-added constraint properties
# ---------------------------------------------------------------------------

_WEO_GDP = (
    "ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_Nominal"
)


def _weo_gdp_post(self, path, payload, *, deadline=None):
    """The dev graph's WEO GDP node: the import tool listed `sourceIndicatorCode`, which
    carries two codes, beside the two declared constraint properties."""
    return {
        "data": {
            _WEO_GDP: {
                "arcs": {
                    "constraintProperties": {
                        "nodes": [
                            {"dcid": "fixedCapitalConsumptionBasis"},
                            {"dcid": "productTaxBasis"},
                            {"dcid": "sourceIndicatorCode"},
                        ]
                    },
                    "sourceIndicatorCode": {"nodes": [{"value": "NGDP"}, {"value": "NGDPD"}]},
                }
            }
        }
    }


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(
            lambda client: {
                prop: tuple(n.get("dcid") or n.get("value") for n in data["nodes"])
                for prop, data in client.node_arcs_batch([_WEO_GDP])[_WEO_GDP].items()
            },
            id="node_arcs_batch",
        ),
        pytest.param(
            lambda client: client.node_arcs_many(
                [_WEO_GDP], properties=["constraintProperties", "sourceIndicatorCode"]
            )[_WEO_GDP],
            id="node_arcs_many",
        ),
        pytest.param(
            lambda client: client.node_arc_values(
                _WEO_GDP, properties=["constraintProperties", "sourceIndicatorCode"]
            ),
            id="node_arc_values",
        ),
    ],
)
def test_forward_arc_reads_drop_the_import_added_source_indicator_code(monkeypatch, read) -> None:
    """Every forward-arc read serves the declared constraint list and keeps the code's own
    multi-valued arc."""
    monkeypatch.setattr(GraphClient, "post", _weo_gdp_post)
    client = GraphClient(base_url="http://fake/core/api")

    assert read(client) == {
        "constraintProperties": ("fixedCapitalConsumptionBasis", "productTaxBasis"),
        "sourceIndicatorCode": ("NGDP", "NGDPD"),
    }
