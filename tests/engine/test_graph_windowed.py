"""Tests date parsing of observation facets in live and fake graph clients."""

from __future__ import annotations

import dc_graph
from search.engine.clients.graph_client import LiveGraphClient
from tests.fixtures import FakeGraph

# Admission drops a facet whose metadata names no provenance.
_FACETS = {
    "f1": {"provenanceId": "dc/base/TestProvenance"},
    "facet-a": {"provenanceId": "dc/base/TestProvenance"},
}

# LiveGraphClient: observation_facets parses dates


def test_live_observation_facets_parses_dates(monkeypatch) -> None:
    """An observations array populates `Facet.dates`."""
    sv = "SomeVar"
    entity = "country/TEST"

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "byVariable": {
                sv: {
                    "byEntity": {
                        entity: {
                            "orderedFacets": [
                                {
                                    "facetId": "f1",
                                    "earliestDate": "2018",
                                    "latestDate": "2022",
                                    "obsCount": 5,
                                    "observations": [
                                        {"date": "2018", "value": 1.0},
                                        {"date": "2019", "value": 2.0},
                                        {"date": "2020", "value": 3.0},
                                        {"date": "2021", "value": 4.0},
                                        {"date": "2022", "value": 5.0},
                                    ],
                                }
                            ]
                        }
                    }
                }
            },
            "facets": _FACETS,
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")
    facets = client.observation_facets(stat_var=sv, entity=entity)

    assert len(facets) == 1
    assert facets[0].dates == ("2018", "2019", "2020", "2021", "2022")


def test_live_observation_facets_skips_missing_dates(monkeypatch) -> None:
    """Observation entries without a 'date' field are skipped."""
    sv = "SomeVar"
    entity = "country/TEST"

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "byVariable": {
                sv: {
                    "byEntity": {
                        entity: {
                            "orderedFacets": [
                                {
                                    "facetId": "f1",
                                    "earliestDate": "2020",
                                    "latestDate": "2021",
                                    "obsCount": 2,
                                    "observations": [
                                        {"date": "2020", "value": 1.0},
                                        {"value": 9.9},  # Missing dates are skipped.
                                        {"date": "2021", "value": 2.0},
                                    ],
                                }
                            ]
                        }
                    }
                }
            },
            "facets": _FACETS,
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")
    facets = client.observation_facets(stat_var=sv, entity=entity)

    assert len(facets) == 1
    assert facets[0].dates == ("2020", "2021")


# LiveGraphClient: an incomplete facet keeps its known dates


def test_live_observation_facets_preserve_known_dates_from_an_incomplete_facet(
    monkeypatch,
) -> None:
    sv = "SomeVar"
    entity = "country/TEST"

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "byVariable": {
                sv: {
                    "byEntity": {
                        entity: {
                            "orderedFacets": [
                                {
                                    "facetId": "facet-a",
                                    "obsCount": 1,
                                    "observations": [{"date": "2020", "value": 1.0}],
                                },
                                {"facetId": "facet-a", "obsCount": 2},
                            ]
                        }
                    }
                }
            },
            "facets": _FACETS,
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")
    facets = client.observation_facets(stat_var=sv, entity=entity, needs_dates=True)

    assert [d for f in facets for d in f.dates] == ["2020"]


# FakeGraph: observation_facets populates dates from observations array


def test_fake_observation_facets_populates_dates_from_observations_array() -> None:
    obs_fixture = {
        "testSV|testEntity": [
            {
                "earliestDate": "2018",
                "latestDate": "2022",
                "obsCount": 3,
                "observations": [
                    {"date": "2018", "value": 1.0},
                    {"date": "2020", "value": 2.0},
                    {"date": "2022", "value": 3.0},
                ],
            }
        ]
    }
    g = FakeGraph(obs=obs_fixture, nodes={}, detect={}, resolve={})
    facets = g.observation_facets(stat_var="testSV", entity="testEntity")
    assert len(facets) == 1
    assert facets[0].dates == ("2018", "2020", "2022")


def test_fake_observation_facets_populates_dates_from_dates_key() -> None:
    """Fixture entry with explicit 'dates' key takes precedence over observations."""
    obs_fixture = {
        "testSV|testEntity": [
            {
                "earliestDate": "2015",
                "latestDate": "2019",
                "obsCount": 3,
                "dates": ["2015", "2017", "2019"],
            }
        ]
    }
    g = FakeGraph(obs=obs_fixture, nodes={}, detect={}, resolve={})
    facets = g.observation_facets(stat_var="testSV", entity="testEntity")
    assert facets[0].dates == ("2015", "2017", "2019")


def test_fake_observation_facets_empty_dates_when_no_observations_key() -> None:
    """Fixtures without dates or observations yield an empty date list."""
    g = FakeGraph()
    facets = g.observation_facets(
        stat_var="FertilityRate_Person_Female",
        entity="country/ETH",
    )
    assert len(facets) == 1
    assert facets[0].dates == ()  # existing fixture has no dates or observations key
