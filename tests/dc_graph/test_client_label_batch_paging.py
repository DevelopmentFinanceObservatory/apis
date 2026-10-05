"""Pins label/type batching and halving recovery when raw provenance rows exceed the graph page cap.
Small chunks exercise multi-request paths without large fixtures."""

import pytest

from dc_graph import GraphClient, GraphReadError


def _name_body(dcids: list[str]) -> dict:
    return {
        "data": {
            dcid: {"arcs": {"name": {"nodes": [{"value": f"Label-{dcid}"}]}}} for dcid in dcids
        }
    }


def _type_body(dcids: list[str]) -> dict:
    return {
        "data": {
            dcid: {"arcs": {"typeOf": {"nodes": [{"dcid": f"Type-{dcid}"}]}}} for dcid in dcids
        }
    }


# ---------------------------------------------------------------------------
# node_labels_batch
# ---------------------------------------------------------------------------


def test_node_labels_batch_miss_list_larger_than_one_chunk_issues_multiple_requests(
    monkeypatch,
) -> None:
    posted: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        posted.append(chunk)
        return _name_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_arc_batch_chunk_size=2)

    dcids = ["a/1", "a/2", "a/3", "a/4", "a/5"]
    result = client.node_labels_batch(dcids)

    assert len(posted) == 3, "5 dcids at chunk size 2 must issue 3 requests"
    assert sorted(posted) == [["a/1", "a/2"], ["a/3", "a/4"], ["a/5"]], (
        "chunks must partition input in order; they post concurrently, in any order"
    )
    assert result == {dcid: f"Label-{dcid}" for dcid in dcids}


def test_node_labels_batch_chunk_reporting_continuation_raises(monkeypatch) -> None:

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        body = _name_body(chunk)
        body["nextToken"] = "tok1"
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_arc_batch_chunk_size=2)

    with pytest.raises(GraphReadError, match="continuation"):
        client.node_labels_batch(["a/1", "a/2"])


def test_node_labels_batch_chunk_continuation_recovers_by_halving(monkeypatch) -> None:
    """A four-dcid chunk reports a continuation. Two-dcid halves fit and recover the full result."""
    posted: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        posted.append(chunk)
        body = _name_body(chunk)
        if len(chunk) > 2:
            body["nextToken"] = "tok1"
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_arc_batch_chunk_size=4)

    dcids = ["a/1", "a/2", "a/3", "a/4"]
    result = client.node_labels_batch(dcids)

    assert result == {dcid: f"Label-{dcid}" for dcid in dcids}
    assert posted == [dcids, ["a/1", "a/2"], ["a/3", "a/4"]], (
        "the full chunk is posted once, then each half once after the continuation"
    )


def test_node_labels_batch_no_name_arc_omitted_without_error_or_retry(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return {"data": {"a/1": {"arcs": {}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_arc_batch_chunk_size=2)

    result = client.node_labels_batch(["a/1"])

    assert result == {}
    assert len(calls) == 1, "an unnamed node must not trigger a retry"


def test_node_labels_batch_preserves_input_order_across_chunks_and_cache_warm_state(
    monkeypatch,
) -> None:

    def fake_post(self, path, payload, *, deadline=None):
        return _name_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", single_arc_batch_chunk_size=2, label_cache_size=100
    )

    dcids = ["a/1", "a/2", "a/3", "a/4", "a/5"]
    cold = client.node_labels_batch(dcids)  # all misses, split across 3 chunks
    warm = client.node_labels_batch(dcids)  # all hits, no chunking at all
    assert list(cold.keys()) == dcids
    assert list(warm.keys()) == dcids
    assert cold == warm


def test_node_labels_batch_second_call_same_dcids_issues_no_request(monkeypatch) -> None:
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return _name_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", single_arc_batch_chunk_size=2, label_cache_size=100
    )

    dcids = ["a/1", "a/2", "a/3"]
    client.node_labels_batch(dcids)
    calls_after_first = call_count[0]
    assert calls_after_first > 0

    client.node_labels_batch(dcids)
    assert call_count[0] == calls_after_first, "a fully warm cache must issue no POST"


# ---------------------------------------------------------------------------
# node_types_batch
# ---------------------------------------------------------------------------


def test_node_types_batch_miss_list_larger_than_one_chunk_issues_multiple_requests(
    monkeypatch,
) -> None:
    posted: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        posted.append(chunk)
        return _type_body(chunk)

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_arc_batch_chunk_size=2)

    dcids = ["a/1", "a/2", "a/3", "a/4", "a/5"]
    result = client.node_types_batch(dcids)

    assert len(posted) == 3
    assert sorted(posted) == [["a/1", "a/2"], ["a/3", "a/4"], ["a/5"]]
    assert result == {dcid: f"Type-{dcid}" for dcid in dcids}


def test_node_types_batch_chunk_reporting_continuation_raises(monkeypatch) -> None:

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        body = _type_body(chunk)
        body["nextToken"] = "tok1"
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_arc_batch_chunk_size=2)

    with pytest.raises(GraphReadError, match="continuation"):
        client.node_types_batch(["a/1", "a/2"])


def test_node_types_batch_chunk_continuation_recovers_by_halving(monkeypatch) -> None:
    """A 241-dcid `->typeOf` chunk can exceed the 500-row cap through duplicate provenance rows."""
    posted: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        posted.append(chunk)
        body = _type_body(chunk)
        if len(chunk) > 2:
            body["nextToken"] = "tok1"
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_arc_batch_chunk_size=4)

    dcids = ["a/1", "a/2", "a/3", "a/4"]
    result = client.node_types_batch(dcids)

    assert result == {dcid: f"Type-{dcid}" for dcid in dcids}
    assert posted == [dcids, ["a/1", "a/2"], ["a/3", "a/4"]]


def test_node_types_batch_no_type_of_arc_omitted_without_error_or_retry(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return {"data": {"a/1": {"arcs": {}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_arc_batch_chunk_size=2)

    result = client.node_types_batch(["a/1"])

    assert result == {}
    assert len(calls) == 1


def test_node_types_batch_preserves_input_order_across_chunks_and_cache_warm_state(
    monkeypatch,
) -> None:

    def fake_post(self, path, payload, *, deadline=None):
        return _type_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", single_arc_batch_chunk_size=2, type_cache_size=100
    )

    dcids = ["a/1", "a/2", "a/3", "a/4", "a/5"]
    cold = client.node_types_batch(dcids)
    warm = client.node_types_batch(dcids)
    assert list(cold.keys()) == dcids
    assert list(warm.keys()) == dcids
    assert cold == warm


def test_node_types_batch_second_call_same_dcids_issues_no_request(monkeypatch) -> None:
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return _type_body(list(payload["nodes"]))

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", single_arc_batch_chunk_size=2, type_cache_size=100
    )

    dcids = ["a/1", "a/2", "a/3"]
    client.node_types_batch(dcids)
    calls_after_first = call_count[0]
    assert calls_after_first > 0

    client.node_types_batch(dcids)
    assert call_count[0] == calls_after_first
