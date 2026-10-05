"""Tests `node_labels_batch` caching, input order, and error handling with a fake transport."""

import pytest

from dc_graph import GraphClient, GraphReadError
from graph_generation.stamp import UNKNOWN_KEY


def test_node_labels_batch_cache_hit_skips_post(monkeypatch) -> None:
    """When all requested dcids are already in _label_cache, no post is issued."""
    post_called = []

    def fake_post(self, path, payload, *, deadline=None):
        post_called.append(payload)
        return {}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", label_cache_size=100)
    # Seed the cache directly.
    client._label_cache[(UNKNOWN_KEY, "country/KEN")] = "Kenya"
    client._label_cache[(UNKNOWN_KEY, "country/ETH")] = "Ethiopia"

    result = client.node_labels_batch(["country/KEN", "country/ETH"])
    assert result == {"country/KEN": "Kenya", "country/ETH": "Ethiopia"}
    assert post_called == [], "No POST should be issued for fully cached inputs"


def test_node_labels_batch_partial_cache_posts_only_misses(monkeypatch) -> None:
    """Cached dcids are not re-fetched; only misses appear in the POST nodes list."""
    posted_nodes: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        posted_nodes.append(list(payload.get("nodes", [])))
        # Return a fake label for the one miss.
        return {"data": {"country/DEU": {"arcs": {"name": {"nodes": [{"value": "Germany"}]}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", label_cache_size=100)
    client._label_cache[(UNKNOWN_KEY, "country/KEN")] = "Kenya"

    result = client.node_labels_batch(["country/KEN", "country/DEU"])
    assert result == {"country/KEN": "Kenya", "country/DEU": "Germany"}
    assert len(posted_nodes) == 1
    assert posted_nodes[0] == ["country/DEU"], "POST must include only the miss"
    # Miss result must be written to cache.
    assert client._label_cache.get((UNKNOWN_KEY, "country/DEU")) == "Germany"


def test_node_labels_batch_preserves_input_order_on_partial_cache(monkeypatch) -> None:
    """Result keys follow input order when cache hits and misses interleave."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                dcid: {"arcs": {"name": {"nodes": [{"value": f"Label-{dcid}"}]}}}
                for dcid in payload.get("nodes", [])
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", label_cache_size=100)
    # Seed a cache hit that appears in the MIDDLE of the input order.
    client._label_cache[(UNKNOWN_KEY, "country/KEN")] = "Kenya"

    dcids = ["country/AAA", "country/KEN", "country/ZZZ"]
    result = client.node_labels_batch(dcids)

    # Returned dict must follow the input order, not [hit, ...misses].
    assert list(result.keys()) == dcids
    assert result == {
        "country/AAA": "Label-country/AAA",
        "country/KEN": "Kenya",
        "country/ZZZ": "Label-country/ZZZ",
    }


def test_node_labels_batch_order_stable_across_cache_warm_state(monkeypatch) -> None:
    """The same input yields the same key order whether or not the cache is warm."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                dcid: {"arcs": {"name": {"nodes": [{"value": f"Label-{dcid}"}]}}}
                for dcid in payload.get("nodes", [])
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", label_cache_size=100)

    dcids = ["a/1", "b/2", "c/3"]
    cold = client.node_labels_batch(dcids)  # all misses
    warm = client.node_labels_batch(dcids)  # all hits
    assert list(cold.keys()) == dcids
    assert list(warm.keys()) == dcids
    assert cold == warm


def test_node_labels_batch_writes_found_labels_to_cache(monkeypatch) -> None:
    """Found labels are written to _label_cache so subsequent calls skip the network."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        dcid = payload["nodes"][0]
        return {"data": {dcid: {"arcs": {"name": {"nodes": [{"value": f"Label-{dcid}"}]}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", label_cache_size=100)

    # First call issues a POST.
    result1 = client.node_labels_batch(["country/XYZ"])
    assert result1 == {"country/XYZ": "Label-country/XYZ"}
    assert call_count[0] == 1

    # Second call is a cache hit: no POST.
    result2 = client.node_labels_batch(["country/XYZ"])
    assert result2 == {"country/XYZ": "Label-country/XYZ"}
    assert call_count[0] == 1, "Cache hit must suppress the second POST"


def test_node_labels_batch_none_not_cached(monkeypatch) -> None:
    """Absent nodes (None result) are not cached; a second call re-issues the POST."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return {"data": {}}  # node absent

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", label_cache_size=100)

    result1 = client.node_labels_batch(["absent/NODE"])
    result2 = client.node_labels_batch(["absent/NODE"])

    assert result1 == {}
    assert result2 == {}
    assert call_count[0] == 2, "Absent nodes must not be cached; each call should POST"


def test_node_labels_batch_skips_valueless_name_node_beside_valued(monkeypatch) -> None:
    """A dcid whose `->name` nodes carry a value-less entry beside a valued
    one still returns the valued label, rather than raising `IndexError`.
    """

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "country/AAA": {"arcs": {"name": {"nodes": [{"dcid": "x"}]}}},
                "country/BBB": {"arcs": {"name": {"nodes": [{"value": "B"}]}}},
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.node_labels_batch(["country/AAA", "country/BBB"])
    assert result == {"country/BBB": "B"}


def test_node_labels_batch_error_before_cache_write(monkeypatch) -> None:
    """A GraphReadError from post propagates without any cache write."""

    def fake_post(self, path, payload, *, deadline=None):
        raise GraphReadError("simulated 503", upstream_status=503)

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", label_cache_size=100)

    with pytest.raises(GraphReadError):
        client.node_labels_batch(["country/KEN"])

    assert (UNKNOWN_KEY, "country/KEN") not in client._label_cache
