"""Tests for recorded facet data, graph merging, and recorder protocol signatures."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import pytest

import tests.fixtures.record as record
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.clients.sdmx import EngineSdmxClient
from search.eval.cases import Case, validate_case
from search.eval.runner import build_request
from tests.engine._fake_sdmx import FakeSdmxClient, _facet_from_dict, _parse_role_tuple
from tests.fixtures.record import (
    _facet_to_dict,
    _facet_to_graph_obs_dict,
    _RecordingGraph,
    _RecordingSdmx,
    _role_tuple_key,
)


def test_facet_to_dict_round_trips_through_facet_from_dict():
    """Serialising and reloading a Facet preserves dates, values, and summary fields."""
    facet = Facet(
        earliest_date="2020",
        latest_date="2022",
        obs_count=2,
        dates=("2020", "2022"),
        values=(1.0, None),
        provenance_id="dc/base/Src",
        measurement_method="dcAggregate/Census",
        unit="testUnit",
        scaling_factor="1000",
        facet_id="facet1",
    )

    rebuilt = _facet_from_dict(_facet_to_dict(facet))

    assert rebuilt == facet


def test_facet_to_dict_round_trips_with_no_optional_fields():
    facet = Facet(
        earliest_date="2020", latest_date="2020", obs_count=1, dates=("2020",), values=(5.0,)
    )

    rebuilt = _facet_from_dict(_facet_to_dict(facet))

    assert rebuilt == facet


def test_recording_sdmx_availability_passes_deadline_and_keys_canonical_variables(monkeypatch):
    """Record the caller deadline and canonicalize duplicate, unordered StatVars in the key."""
    _no_sleep(monkeypatch)

    class _LiveSdmx:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def availability_values(self, **kwargs):
            self.calls.append(kwargs)
            return ["country/KEN"]

    live = _LiveSdmx()
    availability: dict[str, list[str]] = {}
    recorder = _RecordingSdmx(live, obs={}, availability=availability)

    assert recorder.availability_values(
        variables=["Count_Worker", "Count_Person", "Count_Worker"],
        component="recipient",
        deadline=123.0,
    ) == ["country/KEN"]

    assert live.calls == [
        {
            "variables": ["Count_Worker", "Count_Person", "Count_Worker"],
            "component": "recipient",
            "filters": None,
            "time_period": None,
            "max_values": None,
            "max_bytes": None,
            "deadline": 123.0,
            "read_seconds": None,
        }
    ]
    assert availability == {"recipient|variables=Count_Person,Count_Worker": ["country/KEN"]}


class _FakeUnderlyingGraph:
    """Graph double keyed by (stat_var, entity)."""

    def __init__(self, has_data: set[tuple[str, str]]):
        self._has_data = has_data
        self.calls: list[tuple[str, str]] = []
        self.batch_calls: list[tuple[list[str], list[str], bool, int | None]] = []

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        self.calls.append((stat_var, entity))
        if (stat_var, entity) in self._has_data:
            return [Facet(earliest_date="2020", latest_date="2020", obs_count=1)]
        return []

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        self.batch_calls.append((list(stat_vars), list(entities), needs_dates, max_bytes))
        return {
            sv: {
                entity: (
                    [Facet(earliest_date="2020", latest_date="2020", obs_count=1)]
                    if (sv, entity) in self._has_data
                    else []
                )
                for entity in entities
            }
            for sv in stat_vars
        }


def _recording_graph(fake: _FakeUnderlyingGraph) -> _RecordingGraph:
    return _RecordingGraph(fake, nodes={}, obs={}, detect={}, resolve={})


def _no_sleep(monkeypatch) -> list[float]:
    """Record sleep durations while suppressing delays."""
    calls: list[float] = []
    monkeypatch.setattr(record.time, "sleep", lambda s: calls.append(s))
    return calls


def test_observation_facets_batch_issues_one_underlying_call_for_any_size(monkeypatch):
    """Any StatVar/entity batch makes one underlying call and one sleep."""
    sleeps = _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph({("v1", "e1")})
    graph = _recording_graph(fake)

    result = graph.observation_facets_batch(["v1", "v2", "v3"], ["e1", "e2", "e3", "e4"])

    assert len(fake.batch_calls) == 1
    assert fake.calls == []
    assert len(sleeps) == 1
    assert set(result.keys()) == {"v1", "v2", "v3"}


def test_observation_facets_batch_empty_stat_vars_returns_empty_without_calling(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph(set())
    graph = _recording_graph(fake)

    result = graph.observation_facets_batch([], ["e1"])

    assert result == {}
    assert fake.batch_calls == []
    assert sleeps == []


def test_observation_facets_batch_empty_entities_returns_empty_without_calling(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph(set())
    graph = _recording_graph(fake)

    result = graph.observation_facets_batch(["v1"], [])

    assert result == {}
    assert fake.batch_calls == []
    assert sleeps == []


def test_observation_facets_batch_records_each_pair_as_graph_obs_dict(monkeypatch):
    """Graph observations store summary fields without dates or values."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph({("v1", "e1")})
    graph = _recording_graph(fake)

    graph.observation_facets_batch(["v1"], ["e1", "e2"])

    facet = Facet(earliest_date="2020", latest_date="2020", obs_count=1)
    assert graph._obs == {
        "v1|e1": [_facet_to_graph_obs_dict(facet)],
        "v1|e2": [],
    }
    assert "dates" not in graph._obs["v1|e1"][0]
    assert "values" not in graph._obs["v1|e1"][0]


def test_observation_facets_batch_does_not_overwrite_existing_key(monkeypatch):
    """Batch recording preserves an existing curated key."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph({("v1", "e1")})
    curated = [{"earliestDate": "hand-curated", "latestDate": "hand-curated", "obsCount": 99}]
    graph = _RecordingGraph(fake, nodes={}, obs={"v1|e1": curated}, detect={}, resolve={})

    graph.observation_facets_batch(["v1"], ["e1"])

    assert graph._obs["v1|e1"] == curated


def test_facet_to_graph_obs_dict_emits_dates_and_values_when_the_facet_carries_them():
    """Date-aware calls record the Facet's dates and values."""
    facet = Facet(
        earliest_date="2020",
        latest_date="2022",
        obs_count=2,
        dates=["2020", "2022"],
        values=[1.0, None],
    )

    entry = _facet_to_graph_obs_dict(facet)

    assert entry["dates"] == ["2020", "2022"]
    assert entry["values"] == [1.0, None]


def test_facet_to_graph_obs_dict_omits_dates_and_values_when_the_facet_has_none():
    """Summary-only facets omit dates and values."""
    facet = Facet(earliest_date="2020", latest_date="2020", obs_count=1)

    entry = _facet_to_graph_obs_dict(facet)

    assert "dates" not in entry
    assert "values" not in entry


def test_facet_to_graph_obs_dict_emits_facet_id_when_present():
    """Recorded facetId supports matching the same facet on later reads."""
    facet = Facet(earliest_date="2020", latest_date="2020", obs_count=1, facet_id="f1")

    entry = _facet_to_graph_obs_dict(facet)

    assert entry["facetId"] == "f1"


def test_facet_to_graph_obs_dict_omits_facet_id_when_absent():
    facet = Facet(earliest_date="2020", latest_date="2020", obs_count=1)

    entry = _facet_to_graph_obs_dict(facet)

    assert "facetId" not in entry


class _OneShotGraph:
    """Return fixed facets regardless of request arguments."""

    def __init__(self, facets: list[Facet]):
        self._facets = facets

    def observation_facets(self, *, stat_var: str, entity: str, needs_dates: bool = False):
        return self._facets


def test_date_less_stored_entry_is_upgraded_in_place_by_a_later_date_bearing_call(monkeypatch):
    """A later date-aware call adds dates and values when identity and summary fields match.

    Existing provenance and summary fields stay unchanged.
    """
    _no_sleep(monkeypatch)
    stored = {
        "earliestDate": "2020",
        "latestDate": "2022",
        "obsCount": 2,
        "provenanceId": "dc/base/Src",
    }
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2020",
                latest_date="2022",
                obs_count=2,
                dates=["2020", "2022"],
                values=[1.0, 2.0],
                provenance_id="dc/base/Src",
            )
        ]
    )
    graph = _RecordingGraph(fake, nodes={}, obs={"v1|e1": [dict(stored)]}, detect={}, resolve={})

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entry = graph._obs["v1|e1"][0]
    assert entry["dates"] == ["2020", "2022"]
    assert entry["values"] == [1.0, 2.0]
    assert entry["earliestDate"] == "2020"
    assert entry["latestDate"] == "2022"
    assert entry["obsCount"] == 2
    assert entry["provenanceId"] == "dc/base/Src"


@pytest.mark.parametrize(
    ("stored", "incoming_facets"),
    [
        pytest.param(
            {
                "earliestDate": "2020",
                "latestDate": "2022",
                "obsCount": 2,
                "provenanceId": "dc/base/Src",
            },
            [
                Facet(
                    earliest_date="1999",
                    latest_date="1999",
                    obs_count=99,
                    dates=["2020", "2022"],
                    values=[1.0, 2.0],
                    provenance_id="dc/base/Src",
                )
            ],
            id="summary-disagrees",
        ),
        pytest.param(
            {
                "earliestDate": "2020",
                "latestDate": "2022",
                "obsCount": 2,
                "provenanceId": "dc/base/Src",
            },
            [
                Facet(
                    earliest_date="2020",
                    latest_date="2022",
                    obs_count=2,
                    dates=["2020", "2022"],
                    values=[1.0, 2.0],
                    provenance_id="dc/base/Other",
                )
            ],
            id="identity-disagrees",
        ),
        pytest.param(
            {"earliestDate": "2020", "latestDate": "2022", "obsCount": 2},
            [
                Facet(
                    earliest_date="2020",
                    latest_date="2022",
                    obs_count=2,
                    dates=["2020", "2022"],
                    values=[1.0, 2.0],
                )
            ],
            id="no-identity",
        ),
        pytest.param(
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "dates": ["2020"],
                "values": [5.0],
            },
            [
                Facet(
                    earliest_date="2020",
                    latest_date="2020",
                    obs_count=1,
                    dates=["2099"],
                    values=[999.0],
                )
            ],
            id="already-dated",
        ),
        pytest.param(
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": "dc/base/Src",
            },
            [
                Facet(
                    earliest_date="2020",
                    latest_date="2020",
                    obs_count=1,
                    dates=["2020"],
                    values=[10.0],
                    provenance_id="dc/base/Src",
                ),
                Facet(
                    earliest_date="2020",
                    latest_date="2020",
                    obs_count=1,
                    dates=["2099"],
                    values=[99.0],
                    provenance_id="dc/base/Src",
                ),
            ],
            id="incoming-identity-tie",
        ),
    ],
)
def test_legacy_entry_refusal_preserves_the_whole_stored_entry(
    monkeypatch, stored, incoming_facets
):
    _no_sleep(monkeypatch)
    fake = _OneShotGraph(incoming_facets)
    graph = _RecordingGraph(fake, nodes={}, obs={"v1|e1": [dict(stored)]}, detect={}, resolve={})

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    assert graph._obs["v1|e1"][0] == stored


def test_two_identity_tied_stored_entries_with_one_incoming_facet_neither_upgraded(monkeypatch):
    """Two legacy entries share identity and summary, but only one facet is available.

    The stored-side tie prevents either entry from claiming its dates.
    """
    _no_sleep(monkeypatch)
    stored = [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
        },
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
        },
    ]
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=1,
                dates=["2020"],
                values=[10.0],
                provenance_id="dc/base/Src",
            )
        ]
    )
    graph = _RecordingGraph(
        fake, nodes={}, obs={"v1|e1": [dict(e) for e in stored]}, detect={}, resolve={}
    )

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entries = graph._obs["v1|e1"]
    assert entries[0] == stored[0]
    assert entries[1] == stored[1]


def test_facet_id_match_replaces_stale_summary_fields_wholesale(monkeypatch):
    """A facetId match replaces the stored entry, refreshing stale summary fields."""
    _no_sleep(monkeypatch)
    stored = {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": "dc/base/Src",
        "facetId": "F1",
    }
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2020",
                latest_date="2021",
                obs_count=2,
                dates=["2020", "2021"],
                values=[10.0, 11.0],
                provenance_id="dc/base/Src",
                facet_id="F1",
            )
        ]
    )
    graph = _RecordingGraph(fake, nodes={}, obs={"v1|e1": [dict(stored)]}, detect={}, resolve={})

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entry = graph._obs["v1|e1"][0]
    assert entry["dates"] == ["2020", "2021"]
    assert entry["values"] == [10.0, 11.0]
    assert entry["latestDate"] == max(entry["dates"])
    assert entry["obsCount"] == len(entry["dates"]) == 2
    assert entry["facetId"] == "F1"


def test_upgrade_applies_and_omits_unit_when_only_the_incoming_facet_carries_it(monkeypatch):
    """A missing stored unit permits matching, but upgrades add only dates and values."""
    _no_sleep(monkeypatch)
    stored = {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": "dc/base/Src",
    }
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=1,
                dates=["2020"],
                values=[5.0],
                provenance_id="dc/base/Src",
                unit="testUnit",
            )
        ]
    )
    graph = _RecordingGraph(fake, nodes={}, obs={"v1|e1": [dict(stored)]}, detect={}, resolve={})

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entry = graph._obs["v1|e1"][0]
    assert entry["dates"] == ["2020"]
    assert entry["values"] == [5.0]
    assert "unit" not in entry


def test_upgrade_handles_mismatched_list_lengths_without_raising(monkeypatch):
    """Identity selects partners. Unpaired entries stay unchanged."""
    _no_sleep(monkeypatch)
    stored_pairable = {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": "dc/base/Src",
    }
    stored_unpaired = {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1}
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=1,
                dates=["2020"],
                values=[1.0],
                provenance_id="dc/base/Src",
            )
        ]
    )
    graph = _RecordingGraph(
        fake,
        nodes={},
        obs={"v1|e1": [dict(stored_pairable), dict(stored_unpaired)]},
        detect={},
        resolve={},
    )

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entries = graph._obs["v1|e1"]
    assert entries[0]["dates"] == ["2020"]
    assert entries[1] == stored_unpaired


def test_same_identity_facets_returned_in_reverse_order_pair_correctly_by_facet_id(monkeypatch):
    """Facet ids preserve pairings when two same-provenance facets return in reverse order."""
    _no_sleep(monkeypatch)
    stored = [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
            "facetId": "F1",
        },
        {
            "earliestDate": "2021",
            "latestDate": "2021",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
            "facetId": "F2",
        },
    ]
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2021",
                latest_date="2021",
                obs_count=1,
                dates=["2021"],
                values=[20.0],
                provenance_id="dc/base/Src",
                facet_id="F2",
            ),
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=1,
                dates=["2020"],
                values=[10.0],
                provenance_id="dc/base/Src",
                facet_id="F1",
            ),
        ]
    )
    graph = _RecordingGraph(
        fake, nodes={}, obs={"v1|e1": [dict(e) for e in stored]}, detect={}, resolve={}
    )

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entries = graph._obs["v1|e1"]
    assert entries[0]["facetId"] == "F1"
    assert entries[0]["dates"] == ["2020"]
    assert entries[0]["values"] == [10.0]
    assert entries[1]["facetId"] == "F2"
    assert entries[1]["dates"] == ["2021"]
    assert entries[1]["values"] == [20.0]


def test_legacy_entry_upgrades_when_exactly_one_of_several_incoming_facets_matches(monkeypatch):
    """One identity match upgrades the entry despite other mismatching facets."""
    _no_sleep(monkeypatch)
    stored = {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": "dc/base/Src",
    }
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2050",
                latest_date="2050",
                obs_count=1,
                dates=["2050"],
                values=[50.0],
                provenance_id="dc/base/Other",
            ),
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=1,
                dates=["2020"],
                values=[10.0],
                provenance_id="dc/base/Src",
            ),
        ]
    )
    graph = _RecordingGraph(fake, nodes={}, obs={"v1|e1": [dict(stored)]}, detect={}, resolve={})

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entry = graph._obs["v1|e1"][0]
    assert entry["dates"] == ["2020"]
    assert entry["values"] == [10.0]


def test_facet_id_claim_leaves_nothing_for_a_same_identity_legacy_sibling(monkeypatch):
    """The facetId entry claims the sole incoming facet, leaving the legacy sibling unchanged."""
    _no_sleep(monkeypatch)
    stored = [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
            "facetId": "F1",
        },
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
        },
    ]
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=1,
                dates=["2020"],
                values=[10.0],
                provenance_id="dc/base/Src",
                facet_id="F1",
            )
        ]
    )
    graph = _RecordingGraph(
        fake, nodes={}, obs={"v1|e1": [dict(e) for e in stored]}, detect={}, resolve={}
    )

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entries = graph._obs["v1|e1"]
    assert entries[0]["dates"] == ["2020"]
    assert entries[0]["values"] == [10.0]
    assert entries[1] == stored[1]


def test_two_stored_entries_sharing_a_facet_id_neither_claims_it(monkeypatch):
    """Duplicate stored facetIds leave both entries unchanged."""
    _no_sleep(monkeypatch)
    stored = [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
            "facetId": "F1",
        },
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Other",
            "facetId": "F1",
        },
    ]
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=1,
                dates=["2020"],
                values=[10.0],
                provenance_id="dc/base/Src",
                facet_id="F1",
            )
        ]
    )
    graph = _RecordingGraph(
        fake, nodes={}, obs={"v1|e1": [dict(e) for e in stored]}, detect={}, resolve={}
    )

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entries = graph._obs["v1|e1"]
    assert entries[0] == stored[0]
    assert entries[1] == stored[1]


def test_identity_tied_entries_with_distinct_summaries_only_the_matching_one_upgrades(
    monkeypatch,
):
    """Only the entry whose summary matches the incoming facet is upgraded."""
    _no_sleep(monkeypatch)
    stored = [
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": "dc/base/Src",
        },
        {
            "earliestDate": "1999",
            "latestDate": "1999",
            "obsCount": 9,
            "provenanceId": "dc/base/Src",
        },
    ]
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="2020",
                latest_date="2020",
                obs_count=1,
                dates=["2020"],
                values=[10.0],
                provenance_id="dc/base/Src",
            )
        ]
    )
    graph = _RecordingGraph(
        fake, nodes={}, obs={"v1|e1": [dict(e) for e in stored]}, detect={}, resolve={}
    )

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    entries = graph._obs["v1|e1"]
    assert entries[0]["dates"] == ["2020"]
    assert entries[0]["values"] == [10.0]
    assert entries[1] == stored[1]


def test_refused_upgrade_prints_the_key_and_reason_to_stderr(monkeypatch, capsys):
    """Refused upgrades print the key and reason to stderr."""
    _no_sleep(monkeypatch)
    stored = {
        "earliestDate": "2020",
        "latestDate": "2022",
        "obsCount": 2,
        "provenanceId": "dc/base/Src",
    }
    fake = _OneShotGraph(
        [
            Facet(
                earliest_date="1999",
                latest_date="1999",
                obs_count=99,
                dates=["2020", "2022"],
                values=[1.0, 2.0],
                provenance_id="dc/base/Src",
            )
        ]
    )
    graph = _RecordingGraph(fake, nodes={}, obs={"v1|e1": [dict(stored)]}, detect={}, resolve={})

    graph.observation_facets(stat_var="v1", entity="e1", needs_dates=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "v1|e1" in captured.err
    assert "disagreeing summary" in captured.err


def test_observation_facets_batch_passes_max_bytes_through(monkeypatch):
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph(set())
    graph = _recording_graph(fake)

    graph.observation_facets_batch(["v1"], ["e1"], max_bytes=12345)

    assert fake.batch_calls[0][3] == 12345


class _FakeUnderlyingNodeArcsThenMany:
    """Return rich arcs, then flattened typeOf/name values for one dcid."""

    def __init__(self, arcs: dict, *, label: str, type_of: str):
        self._arcs = arcs
        self._label = label
        self._type_of = type_of

    def node_arcs(self, dcid: str) -> dict | None:
        return self._arcs

    def node_label(self, dcid: str) -> str | None:
        return self._label

    def node_type(self, dcid: str) -> str | None:
        return self._type_of

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return {dcid: {"typeOf": (self._type_of,), "name": (self._label,)} for dcid in dcids}

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        return dict.fromkeys(dcids, (self._label,))


def test_node_arcs_many_preserves_arcs_recorded_by_node_arcs(monkeypatch):
    """A typeOf/name read preserves arcs recorded by node_arcs."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingNodeArcsThenMany(
        {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}},
        label="Financing Provider",
        type_of="Property",
    )
    graph = _RecordingGraph(fake, nodes={}, obs={}, detect={}, resolve={})

    graph.node_arcs("financingProvider")
    graph.node_arcs_many(["financingProvider"], properties=["typeOf", "name"])

    node = graph._nodes["financingProvider"]
    assert node["arcs"] == {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}}
    assert node["label"] == "Financing Provider"
    assert node["type"] == "Property"


def test_regression_role_declaring_schema_node_keeps_range_includes_after_both_paths_run(
    monkeypatch,
):
    """The rangeIncludes arc survives both node_arcs and node_arcs_many reads.

    node_arcs_many once replaced the whole entry with a bare label and type, so schema nodes
    such as financingProvider lost their arcs and test collection failed with KeyError: 'arcs'.
    """
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingNodeArcsThenMany(
        {
            "rangeIncludes": {
                "nodes": [{"dcid": "Country"}, {"dcid": "DACRecipientUnallocatedByCountry"}]
            }
        },
        label="financingProvider",
        type_of="Property",
    )
    graph = _RecordingGraph(fake, nodes={}, obs={}, detect={}, resolve={})

    graph.node_arcs("financingProvider")
    graph.node_arcs_many(["financingProvider"], properties=["typeOf", "name"])

    assert graph._nodes["financingProvider"]["arcs"]["rangeIncludes"] == {
        "nodes": [{"dcid": "Country"}, {"dcid": "DACRecipientUnallocatedByCountry"}]
    }


class _FakeUnderlyingNodeArcsSequence:
    """Return one queued partial arc set per node_arcs call."""

    def __init__(self, slices: Sequence[dict], *, label: str = "Label", type_of: str = "Type"):
        self._slices = list(slices)
        self._label = label
        self._type_of = type_of

    def node_arcs(self, dcid: str) -> dict | None:
        return self._slices.pop(0)

    def node_label(self, dcid: str) -> str | None:
        return self._label

    def node_type(self, dcid: str) -> str | None:
        return self._type_of


def test_node_arcs_unions_non_overlapping_property_sets_across_calls(monkeypatch):
    """Partial node_arcs reads preserve the union of observed properties."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingNodeArcsSequence(
        [
            {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}},
            {"subClassOf": {"nodes": [{"dcid": "Thing"}]}},
        ]
    )
    graph = _RecordingGraph(fake, nodes={}, obs={}, detect={}, resolve={})

    graph.node_arcs("dcid1")
    graph.node_arcs("dcid1")

    assert graph._nodes["dcid1"]["arcs"] == {
        "rangeIncludes": {"nodes": [{"dcid": "Country"}]},
        "subClassOf": {"nodes": [{"dcid": "Thing"}]},
    }


def test_node_arcs_keeps_a_property_absent_from_a_later_read(monkeypatch):
    """A later empty read preserves the earlier rangeIncludes property."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingNodeArcsSequence(
        [
            {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}},
            {},
        ]
    )
    graph = _RecordingGraph(fake, nodes={}, obs={}, detect={}, resolve={})

    graph.node_arcs("dcid1")
    graph.node_arcs("dcid1")

    assert graph._nodes["dcid1"]["arcs"] == {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}}


def test_node_arcs_unions_a_property_s_own_values_across_calls(monkeypatch):
    """Values for one property are unioned across partial reads."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingNodeArcsSequence(
        [
            {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}},
            {"rangeIncludes": {"nodes": [{"dcid": "Organization"}]}},
        ]
    )
    graph = _RecordingGraph(fake, nodes={}, obs={}, detect={}, resolve={})

    graph.node_arcs("dcid1")
    graph.node_arcs("dcid1")

    assert graph._nodes["dcid1"]["arcs"] == {
        "rangeIncludes": {"nodes": [{"dcid": "Country"}, {"dcid": "Organization"}]}
    }


def test_node_arcs_batch_unions_arcs_instead_of_replacing_them(monkeypatch):
    """Batch reads add arcs to existing entries."""
    _no_sleep(monkeypatch)
    nodes = {
        "dcid1": {
            "label": "L",
            "type": "T",
            "arcs": {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}},
        }
    }
    fake = _FakeUnderlyingNodeGraph({"dcid1": {"subClassOf": {"nodes": [{"dcid": "Thing"}]}}})
    graph = _RecordingGraph(fake, nodes=nodes, obs={}, detect={}, resolve={})

    graph.node_arcs_batch(["dcid1"])

    assert nodes["dcid1"]["arcs"] == {
        "rangeIncludes": {"nodes": [{"dcid": "Country"}]},
        "subClassOf": {"nodes": [{"dcid": "Thing"}]},
    }


class _FakeUnderlyingNodeGraph:
    """Node-arcs batch double keyed by dcid."""

    def __init__(self, arcs_by_dcid: dict[str, dict | None]):
        self._arcs_by_dcid = arcs_by_dcid
        self.batch_calls: list[list[str]] = []

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        self.batch_calls.append(list(dcids))
        return {dcid: self._arcs_by_dcid.get(dcid) for dcid in dcids}

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        return {dcid: f"label:{dcid}" for dcid in dcids}


def test_node_observation_properties_batch_records_underlying_node_arcs(monkeypatch):
    """Record full node arcs and derive observationProperties from the stored node."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingNodeGraph(
        {
            "sv/ROLE_BEARING": {
                "observationProperties": {
                    "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
                }
            },
            "sv/STANDARD": {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}},
        }
    )
    nodes: dict = {}
    graph = _RecordingGraph(fake, nodes=nodes, obs={}, detect={}, resolve={})

    result = graph.node_observation_properties_batch(["sv/ROLE_BEARING", "sv/STANDARD"])

    assert result == {
        "sv/ROLE_BEARING": ("financingProvider", "financingRecipient"),
        "sv/STANDARD": (),
    }
    assert fake.batch_calls == [["sv/ROLE_BEARING", "sv/STANDARD"]]
    assert nodes["sv/ROLE_BEARING"]["arcs"] == {
        "observationProperties": {
            "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
        }
    }
    assert nodes["sv/STANDARD"]["arcs"] == {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}}


def test_node_observation_properties_batch_skips_already_recorded_nodes(monkeypatch):
    """Recorded arcs serve the tuple without another read or sleep."""
    calls = _no_sleep(monkeypatch)
    fake = _FakeUnderlyingNodeGraph({})
    nodes = {
        "sv/ALREADY": {
            "label": "Already",
            "type": "StatisticalVariable",
            "arcs": {"observationProperties": {"nodes": [{"dcid": "role/A"}]}},
        }
    }
    graph = _RecordingGraph(fake, nodes=nodes, obs={}, detect={}, resolve={})

    result = graph.node_observation_properties_batch(["sv/ALREADY"])

    assert result == {"sv/ALREADY": ("role/A",)}
    assert fake.batch_calls == []
    assert calls == [], "no sleep when nothing new needs fetching"


def test_node_observation_properties_batch_fetches_arcs_for_node_recorded_without_arcs(
    monkeypatch,
):
    """A label-only node fetches arcs before deriving its tuple."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingNodeGraph(
        {"sv/LABELED": {"observationProperties": {"nodes": [{"dcid": "financingRecipient"}]}}}
    )
    nodes: dict = {}
    graph = _RecordingGraph(fake, nodes=nodes, obs={}, detect={}, resolve={})
    graph.node_labels_batch(["sv/LABELED"])

    result = graph.node_observation_properties_batch(["sv/LABELED"])

    assert fake.batch_calls == [["sv/LABELED"]]
    assert nodes["sv/LABELED"]["arcs"] == {
        "observationProperties": {"nodes": [{"dcid": "financingRecipient"}]}
    }
    assert result == {"sv/LABELED": ("financingRecipient",)}


def test_entities_with_data_preserves_entity_dcids_input_order(monkeypatch):
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph({("v1", "e2"), ("v1", "e1")})
    graph = _recording_graph(fake)

    result = graph.entities_with_data(["v1"], ["e2", "e1"])

    assert result == ["e2", "e1"]
    assert len(fake.batch_calls) == 1
    assert fake.calls == []


def test_entities_with_data_empty_variable_dcids_returns_empty_without_calling(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph(set())
    graph = _recording_graph(fake)

    result = graph.entities_with_data([], ["e1"])

    assert result == []
    assert fake.calls == []
    assert fake.batch_calls == []
    assert sleeps == []


def test_entities_with_data_empty_entity_dcids_returns_empty_without_calling(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph(set())
    graph = _recording_graph(fake)

    result = graph.entities_with_data(["v1"], [])

    assert result == []
    assert fake.calls == []
    assert fake.batch_calls == []
    assert sleeps == []


def test_entities_with_data_is_union_across_variables(monkeypatch):
    """An entity qualifies when any variable has data."""
    _no_sleep(monkeypatch)
    fake = _FakeUnderlyingGraph({("v1", "e1"), ("v2", "e1"), ("v1", "e2")})
    graph = _recording_graph(fake)

    result = graph.entities_with_data(["v1", "v2"], ["e1", "e2"])

    assert result == ["e1", "e2"]
    assert len(fake.batch_calls) == 1


class _FakeUnderlyingSdmx:
    """Return the same role-series result on every call."""

    def __init__(self, series_result=None):
        self._series_result = series_result if series_result is not None else {}

    def role_series(
        self,
        *,
        variables,
        entities,
        facet_ids,
        time_period=None,
        max_rows,
        max_bytes,
        deadline=None,
    ):
        return self._series_result


def _recording_sdmx(fake, obs=None, availability=None) -> _RecordingSdmx:
    return _RecordingSdmx(
        fake,
        obs if obs is not None else {},
        availability if availability is not None else {},
    )


def test_role_tuple_key_produces_the_string_parse_role_tuple_inverts():
    """Role tuple keys sort segments by role name for parsing."""
    role_tuple = (("recipient", "dcidB"), ("provider", "dcidA"))

    key = _role_tuple_key("v1", role_tuple)

    variable, role_tuple_text = key.split("|", 1)
    assert variable == "v1"
    assert _parse_role_tuple(role_tuple_text) == tuple(sorted(role_tuple))


def test_role_series_writes_facets_under_role_tuple_key_in_facet_to_dict_shape(monkeypatch):
    _no_sleep(monkeypatch)
    role_tuple = (("provider", "p1"), ("recipient", "r1"))
    facet = Facet(
        earliest_date="2020",
        latest_date="2021",
        obs_count=2,
        dates=["2020", "2021"],
        values=[1.0, 2.0],
        provenance_id="dc/base/Src",
        facet_id="F1",
    )
    fake = _FakeUnderlyingSdmx(series_result={"v1": {role_tuple: [facet]}})
    sdmx = _recording_sdmx(fake)

    result = sdmx.role_series(
        variables=["v1"],
        entities={"recipient": ["r1"], "provider": ["p1"]},
        facet_ids={"F1"},
        max_rows=100,
        max_bytes=1000,
    )

    assert result == {"v1": {role_tuple: [facet]}}
    assert sdmx._obs[_role_tuple_key("v1", role_tuple)] == [_facet_to_dict(facet)]


def test_role_series_replaces_entry_when_new_facet_has_longer_history(monkeypatch):
    _no_sleep(monkeypatch)
    role_tuple = (("provider", "p1"), ("recipient", "r1"))
    key = _role_tuple_key("v1", role_tuple)
    short_facet = Facet(
        earliest_date="2021",
        latest_date="2021",
        obs_count=1,
        dates=["2021"],
        values=[2.0],
        facet_id="F1",
    )
    long_facet = Facet(
        earliest_date="2020",
        latest_date="2021",
        obs_count=2,
        dates=["2020", "2021"],
        values=[1.0, 2.0],
        facet_id="F1",
    )
    fake = _FakeUnderlyingSdmx(series_result={"v1": {role_tuple: [long_facet]}})
    sdmx = _recording_sdmx(fake, obs={key: [_facet_to_dict(short_facet)]})

    sdmx.role_series(variables=["v1"], entities={}, facet_ids={"F1"}, max_rows=100, max_bytes=1000)

    assert sdmx._obs[key] == [_facet_to_dict(long_facet)]


def test_role_series_leaves_entry_alone_when_new_facet_has_shorter_history(monkeypatch):
    """A shorter windowed read preserves the fuller stored history."""
    _no_sleep(monkeypatch)
    role_tuple = (("provider", "p1"), ("recipient", "r1"))
    key = _role_tuple_key("v1", role_tuple)
    long_facet = Facet(
        earliest_date="2020",
        latest_date="2021",
        obs_count=2,
        dates=["2020", "2021"],
        values=[1.0, 2.0],
        facet_id="F1",
    )
    short_facet = Facet(
        earliest_date="2021",
        latest_date="2021",
        obs_count=1,
        dates=["2021"],
        values=[2.0],
        facet_id="F1",
    )
    fake = _FakeUnderlyingSdmx(series_result={"v1": {role_tuple: [short_facet]}})
    sdmx = _recording_sdmx(fake, obs={key: [_facet_to_dict(long_facet)]})

    sdmx.role_series(variables=["v1"], entities={}, facet_ids={"F1"}, max_rows=100, max_bytes=1000)

    assert sdmx._obs[key] == [_facet_to_dict(long_facet)]


def test_role_series_merges_equal_length_disjoint_histories_by_date(monkeypatch):
    """Equal-length windows with different dates are merged by date."""
    _no_sleep(monkeypatch)
    role_tuple = (("provider", "p1"), ("recipient", "r1"))
    key = _role_tuple_key("v1", role_tuple)
    stored_facet = Facet(
        earliest_date="2020",
        latest_date="2021",
        obs_count=2,
        dates=["2020", "2021"],
        values=[1.0, 2.0],
        facet_id="F1",
    )
    fresh_facet = Facet(
        earliest_date="2021",
        latest_date="2022",
        obs_count=2,
        dates=["2021", "2022"],
        values=[20.0, 3.0],
        facet_id="F1",
    )
    fake = _FakeUnderlyingSdmx(series_result={"v1": {role_tuple: [fresh_facet]}})
    sdmx = _recording_sdmx(fake, obs={key: [_facet_to_dict(stored_facet)]})

    sdmx.role_series(variables=["v1"], entities={}, facet_ids={"F1"}, max_rows=100, max_bytes=1000)

    assert sdmx._obs[key] == [
        {
            **_facet_to_dict(fresh_facet),
            "earliestDate": "2020",
            "latestDate": "2022",
            "obsCount": 3,
            "dates": ["2020", "2021", "2022"],
            "values": [1.0, 20.0, 3.0],
        }
    ]


def test_role_series_appends_a_different_facet_id_under_the_same_key(monkeypatch):
    _no_sleep(monkeypatch)
    role_tuple = (("provider", "p1"), ("recipient", "r1"))
    key = _role_tuple_key("v1", role_tuple)
    f1 = Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=1,
        dates=["2020"],
        values=[1.0],
        facet_id="F1",
    )
    f2 = Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=1,
        dates=["2020"],
        values=[9.0],
        facet_id="F2",
    )
    fake = _FakeUnderlyingSdmx(series_result={"v1": {role_tuple: [f2]}})
    sdmx = _recording_sdmx(fake, obs={key: [_facet_to_dict(f1)]})

    sdmx.role_series(variables=["v1"], entities={}, facet_ids={"F2"}, max_rows=100, max_bytes=1000)

    assert sdmx._obs[key] == [_facet_to_dict(f1), _facet_to_dict(f2)]


def test_role_series_skips_a_facet_with_no_facet_id(monkeypatch):
    """Only facets with facet_id are stored."""
    _no_sleep(monkeypatch)
    role_tuple = (("provider", "p1"), ("recipient", "r1"))
    key = _role_tuple_key("v1", role_tuple)
    facet = Facet(
        earliest_date="2020", latest_date="2020", obs_count=1, dates=["2020"], values=[1.0]
    )
    fake = _FakeUnderlyingSdmx(series_result={"v1": {role_tuple: [facet]}})
    sdmx = _recording_sdmx(fake)

    sdmx.role_series(variables=["v1"], entities={}, facet_ids={"F1"}, max_rows=100, max_bytes=1000)

    assert sdmx._obs.get(key, []) == []


def test_role_series_replaces_an_earlier_stub(monkeypatch):
    """A matching facetId lets role_series replace a one-point stub with full history.

    Older recordings left "stub": true placeholders in sdmx_obs.json.
    """
    _no_sleep(monkeypatch)
    role_tuple = (("provider", "p1"), ("recipient", "r1"))
    key = _role_tuple_key("v1", role_tuple)
    stub = {
        "earliestDate": "2021",
        "latestDate": "2021",
        "obsCount": 1,
        "dates": ["2021"],
        "values": [5.0],
        "facetId": "F1",
        "stub": True,
    }
    real_facet = Facet(
        earliest_date="2019",
        latest_date="2021",
        obs_count=3,
        dates=["2019", "2020", "2021"],
        values=[1.0, 2.0, 3.0],
        facet_id="F1",
    )
    fake = _FakeUnderlyingSdmx(series_result={"v1": {role_tuple: [real_facet]}})
    sdmx = _recording_sdmx(fake, obs={key: [dict(stub)]})

    sdmx.role_series(variables=["v1"], entities={}, facet_ids={"F1"}, max_rows=100, max_bytes=1000)

    assert sdmx._obs[key] == [_facet_to_dict(real_facet)]


def test_recorded_role_series_round_trips_through_fake_sdmx_client(monkeypatch):
    """A recorded facet round-trips through FakeSdmxClient with its history and metadata."""
    _no_sleep(monkeypatch)
    role_tuple = (("provider", "p1"), ("recipient", "r1"))
    facet = Facet(
        earliest_date="2019",
        latest_date="2021",
        obs_count=3,
        dates=("2019", "2020", "2021"),
        values=(1.0, 2.0, 3.0),
        provenance_id="dc/base/Src",
        unit="testUnit",
        facet_id="F1",
    )
    fake = _FakeUnderlyingSdmx(series_result={"v1": {role_tuple: [facet]}})
    sdmx = _recording_sdmx(fake)

    sdmx.role_series(variables=["v1"], entities={}, facet_ids={"F1"}, max_rows=100, max_bytes=1000)

    observations: dict[str, dict[tuple, list[Facet]]] = {}
    for key, facet_dicts in sdmx._obs.items():
        variable, role_tuple_text = key.split("|", 1)
        parsed_role_tuple = _parse_role_tuple(role_tuple_text)
        observations.setdefault(variable, {})[parsed_role_tuple] = [
            _facet_from_dict(d) for d in facet_dicts
        ]

    fake_client = FakeSdmxClient(observations=observations, availability={})

    series = fake_client.role_series(
        variables=["v1"],
        entities={"recipient": ["r1"], "provider": ["p1"]},
        facet_ids={"F1"},
        max_rows=100,
        max_bytes=1000,
    )
    [replayed] = series["v1"][role_tuple]
    assert replayed.dates == ("2019", "2020", "2021")
    assert replayed.values == (1.0, 2.0, 3.0)
    assert replayed.provenance_id == "dc/base/Src"
    assert replayed.unit == "testUnit"


def _param_shape(method) -> dict[str, tuple[str, bool]]:
    """Return parameter kinds and default flags, excluding self and annotations."""
    params = list(inspect.signature(method).parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    return {p.name: (str(p.kind), p.default is not inspect.Parameter.empty) for p in params}


def _signature_mismatches(
    method_name: str, protocol_method, proxy_method, proxy_name: str
) -> list[str]:
    """Report parameter differences that can make protocol calls fail against the proxy.

    Missing protocol parameters and kind mismatches fail. Defaulted proxy extras and varargs fit.
    """
    protocol_params = _param_shape(protocol_method)
    proxy_params = _param_shape(proxy_method)
    mismatches = []

    for name, (kind, has_default) in protocol_params.items():
        if name not in proxy_params:
            mismatches.append(
                f"{method_name}: protocol declares {name!r} ({kind}) but "
                f"{proxy_name} has no such parameter"
            )
            continue
        proxy_kind, proxy_has_default = proxy_params[name]
        if proxy_kind != kind:
            mismatches.append(
                f"{method_name}: {name!r} is {kind} on the protocol but {proxy_kind} on {proxy_name}"
            )
        elif has_default and not proxy_has_default:
            mismatches.append(
                f"{method_name}: {name!r} is optional on the protocol but required on "
                f"{proxy_name} -- a caller omitting it would fail against the proxy"
            )

    for name, (kind, has_default) in proxy_params.items():
        if kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            continue
        if name not in protocol_params and not has_default:
            mismatches.append(
                f"{method_name}: {proxy_name} requires {name!r} ({kind}) that the "
                "protocol does not declare"
            )

    return mismatches


def test_signature_mismatches_reports_missing_defaulted_protocol_parameter():
    """Callers may pass defaulted protocol parameters, so proxies must accept them."""

    class _Protocol:
        def method(self, *, a: int, b: int = 0) -> None: ...

    class _Proxy:
        def method(self, *, a: int) -> None: ...

    mismatches = _signature_mismatches("method", _Protocol.method, _Proxy.method, "_Proxy")

    assert any("'b'" in m for m in mismatches)


@pytest.mark.parametrize(
    "protocol, proxy",
    [(EngineGraphClient, _RecordingGraph), (EngineSdmxClient, _RecordingSdmx)],
    ids=["graph", "sdmx"],
)
def test_recording_proxy_implements_full_protocol(protocol, proxy):
    """Each recording proxy implements its protocol with call-compatible signatures.

    The check compares parameter names, kinds, and defaults, but ignores annotations and values.
    """
    expected = {name for name in dir(protocol) if not name.startswith("_")}
    actual = {name for name in dir(proxy) if not name.startswith("_")}
    missing = sorted(expected - actual)

    assert not missing, (
        f"{proxy.__name__} is missing {protocol.__name__} method(s): {missing}. "
        f"Add each to tests/fixtures/record.py's {proxy.__name__}, wrapping the "
        "live call and recording its result to the matching fixture dict."
    )

    mismatches = []
    for name in sorted(expected & actual):
        mismatches.extend(
            _signature_mismatches(
                name, getattr(protocol, name), getattr(proxy, name), proxy.__name__
            )
        )

    assert not mismatches, (
        f"{proxy.__name__}'s method signatures have drifted from {protocol.__name__}: "
        + "; ".join(mismatches)
    )


# ---------------------------------------------------------------------------
# the request the recorder posts: build_request(case)
# ---------------------------------------------------------------------------

_ROLE_BEARING_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "FinancialFlow",
    "measured_property": "amount",
    "stat_type": "measuredValue",
    "measurement_qualifier": "Nominal",
}


def _case(variable_text: str, **resubmit) -> Case:
    return validate_case(
        {
            "id": "record-case",
            "query": f"dataset_resubmit: {variable_text}",
            "door": "dataset_resubmit",
            "why": "A dataset_resubmit case for the recorder's request.",
            "resubmit": {
                "variable_text": variable_text,
                "shape": _ROLE_BEARING_SHAPE,
                "slots": [],
                **resubmit,
            },
            "expect": {"definite": {}},
        },
        families={},
    )


def _identity(case: Case) -> dict:
    return build_request(case)["input"]["entries"][0]["identity"]


def test_build_request_carries_posted_origin_refs():
    case = _case("aid disbursed from France", origin_dcids=["country/FRA"])

    assert _identity(case)["entities"]["origin"] == ["country/FRA"]


def test_build_request_carries_canonical_origin_dcids_and_facets():
    """The recorder must replay every semantic input on a refined case,
    including the plural origin spelling and explicit facet bindings."""
    case = _case(
        "aid disbursed from France and Germany",
        origin_dcids=["country/FRA", "country/DEU"],
        facets={"unit": {"state": "value", "ref": "USD"}},
    )

    identity = _identity(case)

    # (#572) build_spec_identity sorts origin dcids as DEU, FRA.
    assert identity["entities"]["origin"] == ["country/DEU", "country/FRA"]
    assert identity["facets"]["unit"]["ref"] == "USD"


def test_build_request_leaves_origin_refs_none_when_the_case_omits_them():
    assert "origin" not in _identity(_case("aid disbursed"))["entities"]


def test_build_request_carries_the_when_slot_window():
    when_slot = {
        "key": {"axis": "when", "property": None, "label": "when"},
        "binding": {
            "kind": "value",
            "value": {
                "ref": None,
                "value_kind": "time_window",
                "time_window": {"start_year": 2015, "end_year": 2024},
                "literal": None,
            },
        },
    }
    case = _case("aid disbursed, 2015 to 2024", slots=[when_slot], origin_dcids=["country/FRA"])

    window = _identity(case)["requested_window"]

    assert (window["kind"], window["start_year"], window["end_year"]) == ("range", 2015, 2024)


def test_build_request_builds_raw_text_input_for_a_raw_text_case():
    case = validate_case(
        {
            "id": "raw-case",
            "query": "aid disbursed to Kenya",
            "why": "A raw-text case for the recorder's request.",
            "expect": {"definite": {}},
        },
        families={},
    )

    assert build_request(case)["input"] == {"kind": "raw_text", "query": "aid disbursed to Kenya"}
