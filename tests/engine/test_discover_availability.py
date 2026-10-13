"""Tests place-scoped taxonomies and shape availability."""

from __future__ import annotations

from search.engine.errors import GraphInfraError
from search.engine.reading_coverage import ObservedYears
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import read_slot_taxonomy
from search.engine.retrieval.member_facts import ArcsMemberFacts
from tests.engine._discover_fixtures import (
    _TEST_SV_NAMESPACE,
    PROP_PURPOSE,
    PROP_RECIPIENT,
    PROP_SCHEME,
    _fake_graph_with_svs,
    _three_constraint_arcs,
)
from tests.fixtures import FakeGraph


def _gender_race_arcs(prop: str, value: str) -> dict:
    """Builds arcs for a Person/count SV with one constraint."""
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop}]},
        prop: {"nodes": [{"dcid": value}]},
    }


class _ProbeFailsGraph:
    """Raises ``GraphInfraError`` on availability probes and delegates other graph calls."""

    def __init__(self, inner: FakeGraph):
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, *, deadline=None):
        raise GraphInfraError("simulated transport error")


class _ProbeCountingGraph:
    """Counts availability probes and delegates all graph calls."""

    def __init__(self, inner: FakeGraph):
        self._inner = inner
        self.observation_facets_batch_calls = 0

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, *, deadline=None):
        self.observation_facets_batch_calls += 1
        return self._inner.observation_facets_batch(
            stat_vars, entities, needs_dates=needs_dates, deadline=deadline
        )


class TestAvailabilityFloor:
    """Place availability scopes taxonomy values.

    Slot keys retain the full-group union."""

    KEN = "country/KEN"
    USA = "country/USA"

    def _kenya_scenario_graph(self, obs: dict | None = None) -> FakeGraph:
        """A Kenya-observed female member shares a five-tuple with US-only members, or the
        members observed as `obs` says."""
        nodes = {
            "gender": {"label": "Gender"},
            "race": {"label": "Race"},
            "Count_Person_Female": {
                "label": "Female population",
                "arcs": _gender_race_arcs("gender", "Female"),
            },
            "Count_Person_Male": {
                "label": "Male population",
                "arcs": _gender_race_arcs("gender", "Male"),
            },
            "Count_Person_WhiteAlone": {
                "label": "White alone population",
                "arcs": _gender_race_arcs("race", "WhiteAlone"),
            },
        }
        obs = obs or {
            f"Count_Person_Female|{self.KEN}": [{"obsCount": 5}],
            f"Count_Person_Male|{self.USA}": [{"obsCount": 5}],
            f"Count_Person_WhiteAlone|{self.USA}": [{"obsCount": 5}],
        }
        return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})

    def _svs(self) -> list[str]:
        return ["Count_Person_Female", "Count_Person_Male", "Count_Person_WhiteAlone"]

    TGO = "country/TGO"

    def _togo_scenario_graph(self) -> FakeGraph:
        """The unconstrained member alone has Togo data."""
        nodes = {
            "gender": {"label": "Gender"},
            "Female": {"label": "Female"},
            "Male": {"label": "Male"},
            "Count_Person": {
                "label": "Population",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                },
            },
            "Count_Person_Female": {
                "label": "Female population",
                "arcs": _gender_race_arcs("gender", "Female"),
            },
            "Count_Person_Male": {
                "label": "Male population",
                "arcs": _gender_race_arcs("gender", "Male"),
            },
        }
        obs = {f"Count_Person|{self.TGO}": [{"obsCount": 137}]}
        return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})

    def test_ungendered_aggregate_starves_gender_taxonomy_but_keeps_slot_key(self):
        """The unconstrained Togo member has data. Both gender labels remain."""
        graph = self._togo_scenario_graph()

        shapes = derive_shapes(
            confirmed_svs=["Count_Person", "Count_Person_Female", "Count_Person_Male"],
            graph=graph,
            place_dcids=[self.TGO],
        )

        assert len(shapes) == 1
        shape = shapes[0]
        taxonomy = shape.slot_taxonomy
        assert taxonomy is not None
        assert not any("gender" in k for k in taxonomy), (
            "gender's taxonomy must be empty -- no Togo-available member asserts it"
        )

        constraint_props = {sk.property_dcid for sk in shape.slot_keys if sk.property_dcid}
        assert "gender" in constraint_props

        assert shape.place_available is True
        assert shape.value_labels is not None
        assert shape.value_labels.get("Female") == "Female"
        assert shape.value_labels.get("Male") == "Male"

    def test_place_unknown_taxonomy_unchanged(self):
        """Without a place filter, taxonomy uses the full member union."""
        graph = self._kenya_scenario_graph()

        shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph)

        assert len(shapes) == 1
        taxonomy = shapes[0].slot_taxonomy
        assert taxonomy is not None
        gender_values = next(v for k, v in taxonomy.items() if "gender" in k)
        race_values = next(v for k, v in taxonomy.items() if "race" in k)
        assert set(gender_values) == {"Female", "Male"}
        assert set(race_values) == {"WhiteAlone"}
        assert shapes[0].member_coverage is None

    def test_kenya_place_scoping_prunes_taxonomy_values_but_keeps_slot_keys(self):
        """Kenya scopes taxonomy to Female and removes race values, while both slot keys remain."""
        graph = self._kenya_scenario_graph()

        shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph, place_dcids=[self.KEN])

        assert len(shapes) == 1
        shape = shapes[0]
        taxonomy = shape.slot_taxonomy
        assert taxonomy is not None

        assert not any("race" in k for k in taxonomy)
        gender_key = next(k for k in taxonomy if "gender" in k)
        assert taxonomy[gender_key] == ["Female"]

        constraint_props = {sk.property_dcid for sk in shape.slot_keys if sk.property_dcid}
        assert "race" in constraint_props
        assert "gender" in constraint_props
        assert shape.member_coverage == {
            "Count_Person_Female": {self.KEN: ObservedYears(first=None, last=None, years=None)},
            "Count_Person_Male": {},
            "Count_Person_WhiteAlone": {},
        }

    def test_member_coverage_carries_each_places_facet_years(self):
        """Each member maps the probed places holding its observations to the first and last
        year its facets there report, and to every year they hold when each carries all its
        dates; a facet with no observations counts for nothing."""
        graph = self._kenya_scenario_graph(
            obs={
                f"Count_Person_Female|{self.KEN}": [
                    {"obsCount": 3, "earliestDate": "2012-06", "latestDate": "2018"},
                    {"obsCount": 2, "earliestDate": "2010", "latestDate": "2016"},
                ],
                f"Count_Person_Female|{self.USA}": [
                    {"obsCount": 0, "earliestDate": "1990", "latestDate": "2030"}
                ],
                f"Count_Person_Male|{self.USA}": [
                    {
                        "obsCount": 2,
                        "earliestDate": "2000",
                        "latestDate": "2020",
                        "dates": ["2000", "2020"],
                    }
                ],
            }
        )

        (shape,) = derive_shapes(
            confirmed_svs=self._svs(), graph=graph, place_dcids=[self.KEN, self.USA]
        )

        assert shape.member_coverage == {
            "Count_Person_Female": {self.KEN: ObservedYears(first=2010, last=2018, years=None)},
            "Count_Person_Male": {
                self.USA: ObservedYears(first=2000, last=2020, years=frozenset({2000, 2020}))
            },
            "Count_Person_WhiteAlone": {},
        }

    def test_three_constraint_group_availability_floor_scopes_the_observed_union(self):
        """Place-scoped taxonomy retains the full three-property slot-key union."""
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(
            confirmed_svs=[sv_dcid],
            graph=graph,
            place_dcids=["country/ETH"],
        )
        shape = shapes[0]
        taxonomy = shape.slot_taxonomy
        assert taxonomy is not None
        assert taxonomy[f"how:{PROP_SCHEME}"] == ["ODAGrants"]
        assert taxonomy[f"how:{PROP_PURPOSE}"] == ["DAC/Health"]

        constraint_props = {sk.property_dcid for sk in shape.slot_keys if sk.property_dcid}
        assert {PROP_SCHEME, PROP_PURPOSE, PROP_RECIPIENT} <= constraint_props

    def test_us_query_keeps_census_axes(self):
        """USA taxonomy keeps Male and WhiteAlone, the values observed there."""
        graph = self._kenya_scenario_graph()

        shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph, place_dcids=[self.USA])

        taxonomy = shapes[0].slot_taxonomy
        assert taxonomy is not None
        race_key = next(k for k in taxonomy if "race" in k)
        assert taxonomy[race_key] == ["WhiteAlone"]
        gender_key = next(k for k in taxonomy if "gender" in k)
        assert taxonomy[gender_key] == ["Male"]

    def test_slot_keys_identical_regardless_of_place_availability(self):
        """Kenya, USA, and unscoped derivation keep identical slot keys."""
        graph = self._kenya_scenario_graph()

        def _constraint_props(place_dcids: list[str] | None) -> set[str]:
            shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph, place_dcids=place_dcids)
            return {sk.property_dcid for sk in shapes[0].slot_keys if sk.property_dcid}

        unscoped = _constraint_props(None)
        assert {"gender", "race"} <= unscoped
        assert _constraint_props([self.KEN]) == unscoped
        assert _constraint_props([self.USA]) == unscoped

    def test_zero_available_members_falls_back_to_unscoped_union(self):
        """A place with no group observations falls back to the unscoped taxonomy."""
        graph = self._kenya_scenario_graph()

        shapes = derive_shapes(
            confirmed_svs=self._svs(),
            graph=graph,
            place_dcids=["country/ZZZ"],
        )

        taxonomy = shapes[0].slot_taxonomy
        assert taxonomy is not None
        gender_values = next(v for k, v in taxonomy.items() if "gender" in k)
        race_values = next(v for k, v in taxonomy.items() if "race" in k)
        assert set(gender_values) == {"Female", "Male"}
        assert set(race_values) == {"WhiteAlone"}

    def test_probe_failure_fails_open_to_unscoped_union(self):
        """A failed availability probe leaves the unscoped taxonomy intact."""
        graph = _ProbeFailsGraph(self._kenya_scenario_graph())

        shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph, place_dcids=[self.KEN])

        assert len(shapes) == 1
        taxonomy = shapes[0].slot_taxonomy
        assert taxonomy is not None
        gender_values = next(v for k, v in taxonomy.items() if "gender" in k)
        race_values = next(v for k, v in taxonomy.items() if "race" in k)
        assert set(gender_values) == {"Female", "Male"}
        assert set(race_values) == {"WhiteAlone"}

    def test_read_slot_taxonomy_direct_available_svs_restriction(self):
        """The direct reader uses only the supplied member facts."""
        graph = self._kenya_scenario_graph()
        shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph)
        shape = shapes[0]

        taxonomy = read_slot_taxonomy(
            slot_keys=shape.slot_keys,
            member_facts={
                sv: ArcsMemberFacts(arcs=arcs)
                for sv, arcs in (shape.sv_arc_facts or {}).items()
                if sv == "Count_Person_Female"
            },
        )
        gender_key = next(k for k in taxonomy if "gender" in k)
        assert taxonomy[gender_key] == ["Female"]
        assert not any("race" in k for k in taxonomy)

    def test_availability_probe_fires_whenever_place_dcids_is_given(self):
        """Supplying ``place_dcids`` triggers one availability probe."""
        graph = _ProbeCountingGraph(self._kenya_scenario_graph())

        shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph, place_dcids=[self.KEN])

        assert len(shapes) == 1
        assert graph.observation_facets_batch_calls == 1


class TestPlaceAvailableStamp:
    """``place_available`` is true, false, or unknown when no probe verdict exists.

    Unknown covers no place, a failed probe, and a donor-entity family.
    """

    KEN = "country/KEN"

    def _person_count_graph(self, *, has_data_at: str | None) -> FakeGraph:
        nodes = {
            "Count_Person": {
                "label": "Population",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                },
            }
        }
        obs = {f"Count_Person|{has_data_at}": [{"obsCount": 5}]} if has_data_at else {}
        return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})

    def test_true_when_group_has_data_at_the_probed_place(self):
        graph = self._person_count_graph(has_data_at=self.KEN)
        shapes = derive_shapes(confirmed_svs=["Count_Person"], graph=graph, place_dcids=[self.KEN])
        assert shapes[0].place_available is True

    def test_false_when_probe_runs_and_finds_nothing(self):
        graph = self._person_count_graph(has_data_at="country/ZZZ")
        shapes = derive_shapes(confirmed_svs=["Count_Person"], graph=graph, place_dcids=[self.KEN])
        assert shapes[0].place_available is False

    def test_none_when_place_is_unknown(self):
        graph = self._person_count_graph(has_data_at=self.KEN)
        shapes = derive_shapes(confirmed_svs=["Count_Person"], graph=graph)
        assert shapes[0].place_available is None

    def test_none_when_the_probe_fails_open(self):
        graph = _ProbeFailsGraph(self._person_count_graph(has_data_at=self.KEN))
        shapes = derive_shapes(confirmed_svs=["Count_Person"], graph=graph, place_dcids=[self.KEN])
        assert shapes[0].place_available is None

    def test_false_for_a_three_constraint_group_probed_and_finding_nothing(self):
        """A three-constraint group with no observations gets ``False``."""
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})
        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph, place_dcids=["country/ETH"])
        assert shapes[0].place_available is False
