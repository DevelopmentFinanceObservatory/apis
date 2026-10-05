"""Tests FakeGraph data queries, including donor probes and error propagation."""

import pytest

from search.engine.errors import GraphInfraError
from tests.fixtures import FakeGraph


def test_detect_svs_known_query() -> None:
    g = FakeGraph()
    detection = g.detect_svs("government health spending")
    candidates = list(detection.candidates)
    svs = [c.dcid for c in candidates]
    assert "dc/topic/HealthcareExpenditure" in svs
    # Recorded cosine scores are descending and distinct from the 1.0 fallback.
    assert all(0.0 < c.score <= 1.0 for c in candidates)
    assert candidates == sorted(candidates, key=lambda c: c.score, reverse=True)
    assert detection.un.result() is not None


def test_detect_svs_unknown_query_returns_empty() -> None:
    g = FakeGraph()
    detection = g.detect_svs("some completely unknown query text")
    assert detection.candidates == ()
    assert detection.un.result() is not None


def test_observation_facets_known_pair() -> None:
    g = FakeGraph()
    facets = g.observation_facets(
        stat_var="Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth",
        entity="country/USA",
    )
    assert len(facets) == 1
    assert facets[0].obs_count == 20
    assert facets[0].earliest_date == "2000"
    assert facets[0].latest_date == "2022"


def test_observation_facets_absent_pair_returns_empty() -> None:
    g = FakeGraph()
    facets = g.observation_facets(
        stat_var="Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth",
        entity="country/NRU",
    )
    assert facets == []


def test_raise_on_call_observation_facets() -> None:
    g = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        g.observation_facets(
            stat_var="Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth",
            entity="country/USA",
        )


def test_raise_on_call_detect_svs() -> None:
    g = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        g.detect_svs("health ODA grants")


# ---------------------------------------------------------------------------
# observation_facets_batch on FakeGraph
# ---------------------------------------------------------------------------


def test_observation_facets_batch_known_pairs() -> None:
    """observation_facets_batch returns per-SV per-entity facet lists."""
    g = FakeGraph()
    sv = "Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth"
    entity = "country/USA"
    result = g.observation_facets_batch([sv], [entity])
    assert sv in result
    assert entity in result[sv]
    facets = result[sv][entity]
    assert len(facets) == 1
    assert facets[0].obs_count == 20


def test_observation_facets_batch_absent_pair_returns_empty_list() -> None:
    """Absent (sv, entity) pair maps to [] inside the returned dict."""
    g = FakeGraph()
    sv = "Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth"
    entity = "country/NRU"  # not in fixture
    result = g.observation_facets_batch([sv], [entity])
    assert result[sv][entity] == []


def test_observation_facets_batch_cartesian_product() -> None:
    """All (sv, entity) pairs appear in the result regardless of fixture presence."""
    g = FakeGraph()
    sv1 = "Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth"
    sv2 = "NONEXISTENT/SV"
    e1 = "country/USA"
    e2 = "country/NRU"
    result = g.observation_facets_batch([sv1, sv2], [e1, e2])
    assert set(result.keys()) == {sv1, sv2}
    for sv in [sv1, sv2]:
        assert set(result[sv].keys()) == {e1, e2}


def test_observation_facets_batch_raise_on_call() -> None:
    """raise_on_call=True propagates GraphInfraError from observation_facets_batch."""
    g = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        g.observation_facets_batch(
            ["Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth"], ["country/USA"]
        )


def test_observation_facets_batch_donor_probe_only_for_unconfirmed(monkeypatch) -> None:
    """A donor probe applies only to StatVars with no confirmed recipient in the first batch."""
    sv_with_recipient = "sv/RecipientData"
    sv_without_recipient = "sv/DonorOnly"
    recipient = "country/KEN"
    donor = "country/USA"

    obs_fixture = {
        f"{sv_with_recipient}|{recipient}": [
            {"earliestDate": "2010", "latestDate": "2020", "obsCount": 5}
        ],
        f"{sv_without_recipient}|{donor}": [
            {"earliestDate": "2010", "latestDate": "2020", "obsCount": 3}
        ],
        # sv_with_recipient|donor has NO data (would be double-counted if probe fired)
    }

    g = FakeGraph(obs=obs_fixture, nodes={}, detect={}, resolve={})
    result = g.observation_facets_batch([sv_with_recipient, sv_without_recipient], [recipient])
    # A recipient confirmation excludes the StatVar from the donor probe.
    assert result[sv_with_recipient][recipient][0].obs_count == 5
    # sv_without_recipient not confirmed at recipient
    assert result[sv_without_recipient][recipient] == []


def test_entities_with_data_single_variable() -> None:
    """FakeGraph.entities_with_data replays presence from the _obs fixture."""
    sv = "sv/GDP"
    obs = {
        f"{sv}|country/GTM": [{"earliestDate": "1960", "latestDate": "2024", "obsCount": 65}],
        f"{sv}|country/USA": [{"earliestDate": "1960", "latestDate": "2024", "obsCount": 65}],
        # country/VAT has no entry -- no data.
    }
    g = FakeGraph(obs=obs, nodes={}, detect={}, resolve={})
    result = g.entities_with_data([sv], ["country/GTM", "country/VAT", "country/USA"])
    assert result == ["country/GTM", "country/USA"]


def test_entities_with_data_union_across_variables() -> None:
    """A candidate qualifies if it has data for AT LEAST ONE posted variable."""
    sv_a, sv_b = "sv/A", "sv/B"
    obs = {
        f"{sv_a}|country/GTM": [{"earliestDate": "2000", "latestDate": "2020", "obsCount": 3}],
        f"{sv_b}|country/GTM": [{"earliestDate": "2000", "latestDate": "2020", "obsCount": 3}],
        # country/USA only has sv_a, not sv_b -- still qualifies under the union.
        f"{sv_a}|country/USA": [{"earliestDate": "2000", "latestDate": "2020", "obsCount": 3}],
    }
    g = FakeGraph(obs=obs, nodes={}, detect={}, resolve={})
    result = g.entities_with_data([sv_a, sv_b], ["country/GTM", "country/USA"])
    assert result == ["country/GTM", "country/USA"]


def test_entities_with_data_preserves_input_order() -> None:
    """Result order follows the input entity_dcids, not fixture/dict order."""
    sv = "sv/GDP"
    obs = {
        f"{sv}|country/ZZZ": [{"earliestDate": "2000", "latestDate": "2020", "obsCount": 1}],
        f"{sv}|country/AAA": [{"earliestDate": "2000", "latestDate": "2020", "obsCount": 1}],
    }
    g = FakeGraph(obs=obs, nodes={}, detect={}, resolve={})
    result = g.entities_with_data([sv], ["country/ZZZ", "country/AAA"])
    assert result == ["country/ZZZ", "country/AAA"]


def test_entities_with_data_empty_children_is_valid() -> None:
    """No candidate has data -- a valid, informative empty result, not an error."""
    g = FakeGraph(obs={}, nodes={}, detect={}, resolve={})
    result = g.entities_with_data(["sv/GDP"], ["country/GTM", "country/USA"])
    assert result == []


def test_entities_with_data_empty_inputs_return_empty() -> None:
    g = FakeGraph()
    assert g.entities_with_data([], ["country/GTM"]) == []
    assert g.entities_with_data(["sv/GDP"], []) == []


def test_raise_on_call_entities_with_data() -> None:
    g = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        g.entities_with_data(["sv/GDP"], ["country/GTM"])
