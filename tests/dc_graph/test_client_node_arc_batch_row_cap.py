"""Tests recovery from the graph's 500-row `/v2/node` cap.

The fake transport truncates rows mid-node and repeats `nextToken`; the client retries smaller chunks.
"""

import math

import pytest

import dc_graph.client as graph_module
from dc_graph import GraphClient, GraphReadError
from graph_generation.stamp import UNKNOWN_KEY

_CAP_ROWS = 500


def _arcs_with_rows(n: int, *, key: str) -> dict:
    """One synthetic arc property named key with n distinct row entries."""
    return {key: {"nodes": [{"dcid": f"{key}/{i}"} for i in range(n)]}}


def _capped_fake_post(
    row_counts: dict[str, int], calls: list[list[str]], *, default_rows: int, key: str
):
    """Build a fake POST with a 500-row cap and a repeated continuation token.

    Rows follow request order; the cap can split one dcid's arcs and omit later dcids.
    """

    def fake_post(self, path, payload, *, deadline=None):
        dcids = list(payload["nodes"])
        calls.append(dcids)
        rows_per_dcid = [(dcid, row_counts.get(dcid, default_rows)) for dcid in dcids]
        total = sum(n for _, n in rows_per_dcid)
        truncated = total > _CAP_ROWS
        data: dict[str, dict] = {}
        remaining = _CAP_ROWS
        for dcid, n in rows_per_dcid:
            if truncated and remaining <= 0:
                break
            take = n if not truncated else min(n, remaining)
            data[dcid] = {"arcs": _arcs_with_rows(take, key=key)}
            if truncated:
                remaining -= take
        body: dict = {"data": data}
        if truncated:
            body["nextToken"] = "TOK"
        return body

    return fake_post


def test_node_arcs_batch_2778_dcids_all_full_no_suspect_retry(monkeypatch) -> None:
    """2,778 members fit the configured chunks, including 40-row doubled-provenance members."""
    dcids = [f"member/{i}" for i in range(2778)]
    row_counts = {dcid: (40 if i % 10 == 0 else 20) for i, dcid in enumerate(dcids)}
    calls: list[list[str]] = []
    monkeypatch.setattr(
        GraphClient, "post", _capped_fake_post(row_counts, calls, default_rows=20, key="p")
    )
    client = GraphClient(base_url="http://fake/core/api", arcs_cache_size=10000)

    result = client.node_arcs_batch(dcids)

    assert set(result.keys()) == set(dcids)
    for dcid in dcids:
        arcs = result[dcid]
        assert arcs is not None
        assert len(arcs["p"]["nodes"]) == row_counts[dcid]
    expected_chunks = math.ceil(len(dcids) / graph_module._DEFAULT_NODE_ARC_BATCH_CHUNK_SIZE)
    assert len(calls) == expected_chunks, (
        "every chunk must answer in full on its first POST -- an extra call "
        "would mean the suspect retry fired even though nothing was truncated"
    )


def test_node_arcs_batch_straddled_node_returned_and_cached_in_full(monkeypatch) -> None:
    """Ten 21-row fillers split a 300-row node after 290 rows; retries restore both nodes.

    A partial but present page must not be trusted or cached as the node's full arcs.
    """
    fillers = [f"filler/{i}" for i in range(10)]
    dcids = [*fillers, "straddle/node", "tail/node"]
    row_counts = {**dict.fromkeys(fillers, 21), "straddle/node": 300, "tail/node": 20}
    calls: list[list[str]] = []
    monkeypatch.setattr(
        GraphClient, "post", _capped_fake_post(row_counts, calls, default_rows=21, key="p")
    )
    client = GraphClient(base_url="http://fake/core/api", arcs_cache_size=10000)

    result = client.node_arcs_batch(dcids)

    for dcid in dcids:
        arcs = result[dcid]
        assert arcs is not None, f"{dcid} must not be lost to truncation"
        assert len(arcs["p"]["nodes"]) == row_counts[dcid], (
            f"{dcid} must come back with its full row count, not a truncated page"
        )
    assert len(client._arcs_cache[(UNKNOWN_KEY, "straddle/node")]["p"]["nodes"]) == 300, (
        "the cached entry must hold the straddled node's full arcs, not the "
        "290 rows the truncated first page carried"
    )
    assert len(client._arcs_cache[(UNKNOWN_KEY, "tail/node")]["p"]["nodes"]) == 20


def test_node_arcs_batch_lone_dcid_over_cap_raises(monkeypatch) -> None:
    """A single dcid whose own arcs exceed the row cap has nothing smaller
    left to split into, and raises rather than returning a short mapping."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        GraphClient,
        "post",
        _capped_fake_post({"solo/big": 600}, calls, default_rows=20, key="p"),
    )
    client = GraphClient(base_url="http://fake/core/api", arcs_cache_size=10000)

    with pytest.raises(GraphReadError, match="continuation"):
        client.node_arcs_batch(["solo/big"])

    assert (UNKNOWN_KEY, "solo/big") not in client._arcs_cache


def test_node_arcs_batch_one_chunk_exception_propagates_nothing_cached(monkeypatch) -> None:
    """A failure in the second of two chunks propagates and leaves both chunks uncached."""
    good_chunk = [f"good/{i}" for i in range(12)]
    bad_chunk = ["bad/one", *(f"good2/{i}" for i in range(11))]
    dcids = [*good_chunk, *bad_chunk]

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        if "bad/one" in chunk:
            raise GraphReadError("simulated 503 mid-batch", upstream_status=503)
        return {
            "data": {dcid: {"arcs": {"p": {"nodes": [{"dcid": f"v/{dcid}"}]}}} for dcid in chunk}
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", arcs_cache_size=10000)

    with pytest.raises(GraphReadError, match="simulated 503 mid-batch"):
        client.node_arcs_batch(dcids)

    assert client._arcs_cache == {}, "an exception anywhere in the call must poison no cache entry"


def test_node_observation_properties_batch_over_cap_returns_every_node(monkeypatch) -> None:
    """Twelve 50-row nodes exceed the cap; recovery returns every full target list."""
    dcids = [f"sv/{i}" for i in range(12)]
    row_counts = dict.fromkeys(dcids, 50)  # 12 x 50 = 600, over the 500-row cap
    calls: list[list[str]] = []
    monkeypatch.setattr(
        GraphClient,
        "post",
        _capped_fake_post(row_counts, calls, default_rows=50, key="observationProperties"),
    )
    client = GraphClient(base_url="http://fake/core/api")

    result = client.node_observation_properties_batch(dcids)

    assert set(result.keys()) == set(dcids)
    assert all(len(result[d]) == 50 for d in dcids)
