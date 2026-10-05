"""Tests date selection for facet reads and the node-label cache."""

from __future__ import annotations

import time

import pytest

import dc_graph
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.errors import GraphInfraError

# observation_facets select branching (needs_dates parameter)


def test_observation_facets_no_dates_omits_date_key(monkeypatch) -> None:
    """A date-free request omits the date field from `select` and the payload."""
    sv = "SomeVar"
    entity = "country/TEST"
    captured: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        captured.append(payload)
        return {"byVariable": {sv: {"byEntity": {entity: {"orderedFacets": []}}}}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")
    client.observation_facets(stat_var=sv, entity=entity, needs_dates=False)

    assert len(captured) == 1
    body = captured[0]
    assert body["select"] == ["variable", "entity", "facet"]
    assert "date" not in body


def test_observation_facets_with_dates_includes_date_key(monkeypatch) -> None:
    """A dated request selects five fields and sends an empty `date` value."""
    sv = "SomeVar"
    entity = "country/TEST"
    captured: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        captured.append(payload)
        return {"byVariable": {sv: {"byEntity": {entity: {"orderedFacets": []}}}}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")
    client.observation_facets(stat_var=sv, entity=entity, needs_dates=True)

    assert len(captured) == 1
    body = captured[0]
    assert body["select"] == ["variable", "entity", "date", "value", "facet"]
    assert "date" in body
    assert body["date"] == ""


def test_observation_facets_default_omits_date_key(monkeypatch) -> None:
    """Default call (no needs_dates arg) behaves like needs_dates=False."""
    sv = "SomeVar"
    entity = "country/TEST"
    captured: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        captured.append(payload)
        return {"byVariable": {sv: {"byEntity": {entity: {"orderedFacets": []}}}}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")
    client.observation_facets(stat_var=sv, entity=entity)

    assert len(captured) == 1
    assert "date" not in captured[0]


def test_a_dated_facet_read_passes_its_deadline_and_times_out_as_graph_infra_error(
    monkeypatch,
) -> None:
    """The deadline reaches the facet POST, and a read past it raises `GraphInfraError`,
    which the bound-unit gate's dated read falls back from (`convertibility._dated_plan`)."""
    sv = "SomeVar"
    entity = "country/TEST"
    deadlines: list[float | None] = []

    def fake_post(self, path, payload, *, deadline=None):
        deadlines.append(deadline)
        return {"byVariable": {sv: {"byEntity": {entity: {"orderedFacets": []}}}}}

    with monkeypatch.context() as patched:
        patched.setattr(dc_graph.GraphClient, "post", fake_post)
        LiveGraphClient(base="http://fake").observation_facets_batch(
            [sv], [entity], needs_dates=True, deadline=123.0
        )
    assert deadlines == [123.0]

    with pytest.raises(GraphInfraError):
        LiveGraphClient(base="http://fake").observation_facets_batch(
            [sv], [entity], needs_dates=True, deadline=time.monotonic() - 1
        )


# node_label in-process cache


def test_node_label_cache_deduplicates_http_calls(monkeypatch) -> None:
    """Two node_label calls for the same dcid issue only one POST."""
    call_count = 0

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal call_count
        call_count += 1
        dcid = payload["nodes"][0]
        return {"data": {dcid: {"arcs": {"name": {"nodes": [{"value": "Kenya"}]}}}}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    label1 = client.node_label("country/KEN")
    label2 = client.node_label("country/KEN")

    assert label1 == "Kenya"
    assert label2 == "Kenya"
    assert call_count == 1, f"Expected 1 post call but got {call_count}"


def test_node_label_none_result_not_cached(monkeypatch) -> None:
    """A None-returning dcid is not cached; a second call re-issues the HTTP call."""
    call_count = 0

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal call_count
        call_count += 1
        return {"data": {}}  # absent node

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    result1 = client.node_label("absent/NODE")
    result2 = client.node_label("absent/NODE")

    assert result1 is None
    assert result2 is None
    assert call_count == 2, f"Absent node must not be cached; expected 2 calls, got {call_count}"


def test_node_label_cache_separate_dcids_each_hit_once(monkeypatch) -> None:
    """Two different dcids each issue one HTTP call even when interleaved."""
    calls: list[str] = []

    def fake_post(self, path, payload, *, deadline=None):
        dcid = payload["nodes"][0]
        calls.append(dcid)
        return {"data": {dcid: {"arcs": {"name": {"nodes": [{"value": f"Label-{dcid}"}]}}}}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    client.node_label("a/A")
    client.node_label("b/B")
    client.node_label("a/A")
    client.node_label("b/B")

    # Each dcid should hit once despite two calls each
    assert calls.count("a/A") == 1
    assert calls.count("b/B") == 1
