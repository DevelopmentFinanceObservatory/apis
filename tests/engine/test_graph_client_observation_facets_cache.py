"""Tests facet-query caching and `force_full_facets` date-field overrides."""

import pytest

import dc_graph
from search.engine.clients.graph import force_full_facets
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.errors import GraphInfraError


def test_observation_facets_batch_uses_cache(monkeypatch) -> None:
    """Cached (sv, entity, needs_dates) triples skip the POST on a second batch call."""
    sv = "sv/A"
    entity = "country/X"
    post_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        post_count[0] += 1
        return {
            "byVariable": {
                sv: {
                    "byEntity": {
                        entity: {
                            "orderedFacets": [
                                {
                                    "facetId": "f1",
                                    "earliestDate": "2020",
                                    "latestDate": "2022",
                                    "obsCount": 10,
                                }
                            ]
                        }
                    }
                }
            },
            "facets": {"f1": {"provenanceId": "dc/base/TestProvenance"}},
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    r1 = client.observation_facets_batch([sv], [entity])
    r2 = client.observation_facets_batch([sv], [entity])

    assert post_count[0] == 1, "Second batch call must be served from cache"
    assert r1[sv][entity][0].obs_count == r2[sv][entity][0].obs_count == 10


def test_observation_facets_cache_error_does_not_write(monkeypatch) -> None:
    """A GraphInfraError from _post must not write to the obs cache."""

    def fake_post(self, path, payload, *, deadline=None):
        raise GraphInfraError("upstream down", upstream_status=503)

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError):
        client.observation_facets_batch(["sv/A"], ["country/X"])

    assert len(client._client._ours._obs_cache) == 0


def test_observation_facets_single_pair_uses_same_cache(monkeypatch) -> None:
    """observation_facets (single-pair) shares the cache with the batch method."""
    sv = "sv/B"
    entity = "country/Y"
    post_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        post_count[0] += 1
        return {
            "byVariable": {
                sv: {
                    "byEntity": {
                        entity: {
                            "orderedFacets": [
                                {
                                    "facetId": "f1",
                                    "earliestDate": "2015",
                                    "latestDate": "2021",
                                    "obsCount": 7,
                                }
                            ]
                        }
                    }
                }
            },
            "facets": {"f1": {"provenanceId": "dc/base/TestProvenance"}},
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    r_single = client.observation_facets(stat_var=sv, entity=entity)
    r_batch = client.observation_facets_batch([sv], [entity])

    assert post_count[0] == 1, "Batch call must hit the cache seeded by single-pair call"
    assert r_single[0].obs_count == r_batch[sv][entity][0].obs_count == 7


# force_full_facets: needs_dates override for coverage-probe/observations
# cache sharing (see region/pipeline.py's materialise_task wrap)


def test_force_full_facets_upgrades_needs_dates_false_to_true(monkeypatch) -> None:
    """Within force_full_facets(True), a needs_dates=False call still requests
    the full date/value select -- the override changes what gets POSTed, not
    just the cache key."""
    sv, entity = "sv/FF1", "country/FF"
    seen_bodies: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        seen_bodies.append(payload)
        return {"byVariable": {}, "facets": {}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with force_full_facets(True):
        client.observation_facets_batch([sv], [entity], needs_dates=False)

    assert seen_bodies[0]["select"] == ["variable", "entity", "date", "value", "facet"]


def test_force_full_facets_false_is_a_no_op(monkeypatch) -> None:
    """enabled=False changes nothing -- the exact baseline dateless payload."""
    sv, entity = "sv/FF2", "country/FF"
    seen_bodies: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        seen_bodies.append(payload)
        return {"byVariable": {}, "facets": {}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with force_full_facets(False):
        client.observation_facets_batch([sv], [entity], needs_dates=False)

    assert seen_bodies[0]["select"] == ["variable", "entity", "facet"]


def test_force_full_facets_nested_false_overrides_ambient_true(monkeypatch) -> None:
    """A nested `False` scope uses the dateless payload, then the ambient `True` resumes."""
    sv, entity = "sv/FF6", "country/FF"
    seen_bodies: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        seen_bodies.append(payload)
        return {"byVariable": {}, "facets": {}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with force_full_facets(True):
        with force_full_facets(False):
            client.observation_facets_batch([sv], [entity], needs_dates=False)
        # A different (sv, entity) pair to avoid the shared LRU cache masking
        # whether the ambient True actually resumed after the nested scope.
        client.observation_facets_batch([sv + "-b"], [entity], needs_dates=False)

    assert seen_bodies[0]["select"] == ["variable", "entity", "facet"], (
        "nested force_full_facets(False) must force the dateless payload"
    )
    assert seen_bodies[1]["select"] == ["variable", "entity", "date", "value", "facet"], (
        "the ambient force_full_facets(True) must resume after the nested scope exits"
    )


def test_force_full_facets_applies_to_single_pair_observation_facets(monkeypatch) -> None:
    """The override applies uniformly to the single-pair convenience method too
    (it shares the same cache as observation_facets_batch)."""
    sv, entity = "sv/FF3", "country/FF"
    seen_bodies: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        seen_bodies.append(payload)
        return {"byVariable": {}, "facets": {}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with force_full_facets(True):
        client.observation_facets(stat_var=sv, entity=entity, needs_dates=False)

    assert seen_bodies[0]["select"] == ["variable", "entity", "date", "value", "facet"]


def test_force_full_facets_collapses_probe_and_observations_fetch_to_one_post(
    monkeypatch,
) -> None:
    """The scoped probe requests dates and shares the observations fetch's cache key, saving a POST."""
    sv, entity = "sv/FF4", "country/FF"
    post_count = [0]
    seen_bodies: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        post_count[0] += 1
        seen_bodies.append(payload)
        return {
            "byVariable": {
                sv: {
                    "byEntity": {
                        entity: {
                            "orderedFacets": [
                                {
                                    "facetId": "f1",
                                    "earliestDate": "2020",
                                    "latestDate": "2022",
                                    "obsCount": 5,
                                }
                            ]
                        }
                    }
                }
            },
            "facets": {"f1": {"provenanceId": "dc/base/TestProvenance"}},
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    # The coverage probe: a dateless query would normally pass needs_dates=False.
    with force_full_facets(True):
        client.observation_facets_batch([sv], [entity], needs_dates=False)
    # The later observations fetch: _fetch_observations always passes True.
    client.observation_facets_batch([sv], [entity], needs_dates=True)

    assert post_count[0] == 1, "coverage probe and observations fetch must share one POST"
    assert seen_bodies[0]["select"] == ["variable", "entity", "date", "value", "facet"], (
        "the scoped probe's own POST body carries the full field list, not the "
        "dateless select its own needs_dates=False argument would otherwise produce"
    )
    assert seen_bodies[0]["date"] == "", (
        "the scoped probe's own POST body carries the date key the dateless select omits"
    )

    # A distinct (sv, entity) pair -- a fresh cache entry -- makes the same
    # needs_dates=False call with no ambient scope active, to confirm the two
    # facts above are the scope's doing and not this fake's fixed response.
    sv_unscoped = sv + "-unscoped"
    client.observation_facets_batch([sv_unscoped], [entity], needs_dates=False)
    assert seen_bodies[-1]["select"] == ["variable", "entity", "facet"], (
        "outside the scope, needs_dates=False keeps the dateless select"
    )
    assert "date" not in seen_bodies[-1], (
        "outside the scope, needs_dates=False's POST body carries no date key at all"
    )


def test_without_force_full_facets_probe_and_observations_fetch_still_double_fetch(
    monkeypatch,
) -> None:
    """Without the scope, dateless and dated reads use separate cache entries and issue two POSTs."""
    sv, entity = "sv/FF5", "country/FF"
    post_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        post_count[0] += 1
        return {
            "byVariable": {
                sv: {
                    "byEntity": {
                        entity: {
                            "orderedFacets": [
                                {
                                    "facetId": "f1",
                                    "earliestDate": "2020",
                                    "latestDate": "2022",
                                    "obsCount": 5,
                                }
                            ]
                        }
                    }
                }
            },
            "facets": {"f1": {"provenanceId": "dc/base/TestProvenance"}},
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    client.observation_facets_batch([sv], [entity], needs_dates=False)
    client.observation_facets_batch([sv], [entity], needs_dates=True)

    assert post_count[0] == 2, "dateless probe and observations fetch keys must differ"
