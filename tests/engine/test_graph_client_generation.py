"""Checks that graph-generation changes invalidate LiveGraphClient caches."""

from __future__ import annotations

import dc_graph
from graph_generation import FakeGeneration
from search.engine.clients.graph_client import LiveGraphClient
from tests.origin_catalogue.fake_catalogue import FakeCatalogue


def test_obs_cache_hit_under_same_token_miss_after_advance(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        return {
            "facets": {},
            "byVariable": {"sv1": {"byEntity": {"e1": {"orderedFacets": []}}}},
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = LiveGraphClient(base="http://fake", generation=generation)

    client.observation_facets(stat_var="sv1", entity="e1")
    client.observation_facets(stat_var="sv1", entity="e1")
    assert len(calls) == 1

    generation.advance("gen-b")
    client.observation_facets(stat_var="sv1", entity="e1")
    assert len(calls) == 2


def test_detect_cache_hit_under_same_token_miss_after_advance(monkeypatch, bind_catalogue) -> None:
    bind_catalogue(FakeCatalogue(un_system_enabled=False))
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        return {"entities": []}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = LiveGraphClient(base="http://fake", generation=generation)

    client.detect_svs("total population")
    client.detect_svs("total population")
    assert len(calls) == 1

    generation.advance("gen-b")
    client.detect_svs("total population")
    assert len(calls) == 2
