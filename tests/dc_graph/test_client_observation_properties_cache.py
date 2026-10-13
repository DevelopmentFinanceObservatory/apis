"""The observation-properties cache keeps a read of up to its capacity across chunks, and
one member past it evicts the earliest."""

from __future__ import annotations

import dc_graph.client as graph_module
from dc_graph import GraphClient

_CHUNK_SIZE = graph_module._DEFAULT_NODE_ARC_BATCH_CHUNK_SIZE

# A small stand-in for SEARCH_OBS_PROPERTIES_CACHE_SIZE, spanning several chunks so the
# boundary is exercised across POSTs, not just within one.
_MAX_MEMBERS = 4 * _CHUNK_SIZE + 2


def _fake_post_body(dcids: list[str]) -> dict:
    return {
        "data": {
            d: {"arcs": {"observationProperties": {"nodes": [{"dcid": "observationAbout"}]}}}
            for d in dcids
        }
    }


def _client_at_cache_capacity(*, maxsize: int) -> GraphClient:
    """Build a client with the observation-properties cache set to `maxsize`."""
    return GraphClient(base_url="http://fake/core/api", obs_properties_cache_size=maxsize)


def test_a_refresh_at_the_cache_capacity_repeats_with_no_further_requests(monkeypatch) -> None:
    """A refresh exactly at cache capacity leaves every member cached for the second pass."""
    client = _client_at_cache_capacity(maxsize=_MAX_MEMBERS)
    dcids = [f"dcid/{i}" for i in range(_MAX_MEMBERS)]
    posted_chunks: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        posted_chunks.append(chunk)
        return _fake_post_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)

    first = client.node_observation_properties_batch(dcids)
    assert set(first.keys()) == set(dcids)
    assert posted_chunks, "the cold first pass must issue at least one POST"

    posted_chunks.clear()
    second = client.node_observation_properties_batch(dcids)

    assert second == first
    assert posted_chunks == [], (
        "a repeat refresh over a dcid set no larger than the cache's own maxsize "
        f"must never re-POST an already-cached member; got {posted_chunks!r}"
    )


def test_a_refresh_one_over_the_cache_capacity_evicts_the_earliest_member(monkeypatch) -> None:
    """One member over capacity evicts the earliest entry, so rereading it issues a POST."""
    client = _client_at_cache_capacity(maxsize=_MAX_MEMBERS)
    dcids = [f"dcid/{i}" for i in range(_MAX_MEMBERS + 1)]
    posted_chunks: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        posted_chunks.append(chunk)
        return _fake_post_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)

    client.node_observation_properties_batch(dcids)

    posted_chunks.clear()
    oldest = dcids[0]
    client.node_observation_properties_batch([oldest])

    assert posted_chunks == [[oldest]], (
        f"the earliest member must have been evicted to admit the (maxsize + 1)-th "
        f"one, forcing a re-POST; got {posted_chunks!r}"
    )
