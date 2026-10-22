"""Tests place-scoped taxonomies and shape availability."""

from __future__ import annotations

import pytest

from dfo_platform.timings import PhaseTimings, collecting
from origin_catalogue import bound_catalogue
from search.engine.errors import GraphInfraError
from search.engine.reading_coverage import ObservedYears
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import read_slot_taxonomy
from search.engine.retrieval.member_facts import ArcsMemberFacts
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._discover_fixtures import (
    _TEST_SV_NAMESPACE,
    PROP_PURPOSE,
    PROP_RECIPIENT,
    PROP_SCHEME,
    _fake_graph_with_svs,
    _three_constraint_arcs,
)
from tests.engine._fake_coverage import build_coverage_index
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import FakeCatalogue


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
    """The place-availability probe sets `place_available`, `place_units` and `member_coverage`.

    It narrows neither the slot keys nor the slot taxonomy: each holds every confirmed
    member's values, and the status rule reads availability as evidence."""

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

    def test_ungendered_aggregate_with_data_leaves_both_gender_values_in_the_taxonomy(self):
        """Only the unconstrained member has Togo data, and the shape still lists Female and
        Male, with its availability recorded as true for the bind menu's data line."""
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
        gender_values = next(v for k, v in taxonomy.items() if "gender" in k)
        assert set(gender_values) == {"Female", "Male"}

        constraint_props = {sk.property_dcid for sk in shape.slot_keys if sk.property_dcid}
        assert "gender" in constraint_props

        assert shape.place_available is True
        assert shape.value_labels is not None
        assert shape.value_labels.get("Female") == "Female"
        assert shape.value_labels.get("Male") == "Male"

    @pytest.mark.parametrize(
        ("place_dcids", "place_available", "member_coverage"),
        [
            pytest.param(None, None, None, id="no-place"),
            pytest.param(
                [KEN],
                True,
                {
                    "Count_Person_Female": {KEN: ObservedYears(first=None, last=None, years=None)},
                    "Count_Person_Male": {},
                    "Count_Person_WhiteAlone": {},
                },
                id="kenya-female-only",
            ),
            pytest.param(
                [USA],
                True,
                {
                    "Count_Person_Female": {},
                    "Count_Person_Male": {USA: ObservedYears(first=None, last=None, years=None)},
                    "Count_Person_WhiteAlone": {
                        USA: ObservedYears(first=None, last=None, years=None)
                    },
                },
                id="usa-male-and-white",
            ),
        ],
    )
    def test_taxonomy_holds_every_member_value_whatever_the_place(
        self, place_dcids, place_available, member_coverage
    ):
        """The slot taxonomy lists Female, Male and WhiteAlone at a place where only some
        members have data and with no place at all; the probe records which members have
        data in `member_coverage`, and the slot keys stay whole."""
        graph = self._kenya_scenario_graph()

        shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph, place_dcids=place_dcids)

        assert len(shapes) == 1
        shape = shapes[0]
        taxonomy = shape.slot_taxonomy
        assert taxonomy is not None
        gender_values = next(v for k, v in taxonomy.items() if "gender" in k)
        race_values = next(v for k, v in taxonomy.items() if "race" in k)
        assert set(gender_values) == {"Female", "Male"}
        assert set(race_values) == {"WhiteAlone"}

        constraint_props = {sk.property_dcid for sk in shape.slot_keys if sk.property_dcid}
        assert {"gender", "race"} <= constraint_props
        assert shape.place_available is place_available
        assert shape.member_coverage == member_coverage

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

    def test_three_constraint_group_keeps_the_full_slot_key_union_at_a_place(self):
        """A place with data leaves the taxonomy and all three constraint slot keys whole."""
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

    def test_zero_available_members_records_no_availability_and_keeps_the_taxonomy(self):
        """A place with no group observations reads as unavailable and leaves the taxonomy."""
        graph = self._kenya_scenario_graph()

        shapes = derive_shapes(
            confirmed_svs=self._svs(),
            graph=graph,
            place_dcids=["country/ZZZ"],
        )

        assert shapes[0].place_available is False
        taxonomy = shapes[0].slot_taxonomy
        assert taxonomy is not None
        gender_values = next(v for k, v in taxonomy.items() if "gender" in k)
        race_values = next(v for k, v in taxonomy.items() if "race" in k)
        assert set(gender_values) == {"Female", "Male"}
        assert set(race_values) == {"WhiteAlone"}

    def test_probe_failure_leaves_availability_unknown_and_the_taxonomy_whole(self):
        """A failed availability probe leaves `place_available` unknown, so no absence is
        read, and the taxonomy intact."""
        graph = _ProbeFailsGraph(self._kenya_scenario_graph())

        shapes = derive_shapes(confirmed_svs=self._svs(), graph=graph, place_dcids=[self.KEN])

        assert len(shapes) == 1
        assert shapes[0].place_available is None
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


class _ProbeRecordingGraph:
    """Records the StatVars of each availability probe and delegates all graph calls."""

    def __init__(self, inner: FakeGraph):
        self._inner = inner
        self.probed: list[list[str]] = []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, *, deadline=None):
        self.probed.append(list(stat_vars))
        return self._inner.observation_facets_batch(
            stat_vars, entities, needs_dates=needs_dates, deadline=deadline
        )


class TestCoverageAnswersTwoRoleProbe:
    """A candidate coverage holds with a donor and a recipient role answers the place probe
    from coverage, with the asked place bound to the donor; the rest are read from the graph.

    The two-role variable has two facets at Kenya as donor whose sources differ, so a
    source-scoped served-rows read would keep one unit where the graph lists both, and a
    third facet holding Kenya only as recipient, which the graph's facet read never sees."""

    KEN = "country/KEN"
    TWO_ROLE = "Amount_EconomicActivity_TwoRoleProbe"
    ONE_ROLE = "Amount_EconomicActivity_OneRoleProbe"
    SUBJECT = "Amount_EconomicActivity_SubjectProbe"
    SERVED = "provenance/OECDDAC2A"
    OTHER = "provenance/OECDCreditorReportingSystem"
    ROLES = ("financingProvider", "financingRecipient")

    def _arcs(self, measured_property: str, roles: tuple[str, ...]) -> dict:
        arcs: dict = {
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": measured_property}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        if roles:
            arcs["observationProperties"] = {"nodes": [{"dcid": r} for r in roles]}
        return arcs

    def _graph(self) -> FakeGraph:
        """The graph's view: every facet of each variable at Kenya in the first position."""
        nodes = {
            self.TWO_ROLE: {"label": "Two-role", "arcs": self._arcs("amountTwo", self.ROLES)},
            self.ONE_ROLE: {
                "label": "One-role",
                "arcs": self._arcs("amountOne", ("financingRecipient",)),
            },
            self.SUBJECT: {"label": "Subject", "arcs": self._arcs("amountSubject", ())},
        }
        obs = {
            f"{self.TWO_ROLE}|{self.KEN}": [
                {"obsCount": 2, "unit": "USDollar", "facetId": "f-served"},
                {"obsCount": 1, "unit": "Euro", "facetId": "f-other"},
            ],
            f"{self.ONE_ROLE}|{self.KEN}": [{"obsCount": 3, "unit": "USDollar"}],
            f"{self.SUBJECT}|{self.KEN}": [{"obsCount": 4, "unit": "Percent"}],
        }
        return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})

    def _coverage(self):
        return build_coverage_index(
            [
                (self.TWO_ROLE, "f-served", self.KEN, "country/GHA", [2019, 2020]),
                (self.TWO_ROLE, "f-other", self.KEN, "country/TZA", [2020]),
                (self.TWO_ROLE, "f-recipient", "country/FRA", self.KEN, [2020]),
                (self.ONE_ROLE, "f-one", "country/FRA", self.KEN, [2020]),
            ],
            role_names={
                self.TWO_ROLE: RoleNames(donor=self.ROLES[0], recipient=self.ROLES[1]),
                self.ONE_ROLE: RoleNames(donor=None, recipient=self.ROLES[1]),
            },
            facet_metadata={
                "f-served": FacetMetadata(provenance_id=self.SERVED, unit="USDollar"),
                "f-other": FacetMetadata(provenance_id=self.OTHER, unit="Euro"),
                "f-recipient": FacetMetadata(provenance_id=self.SERVED, unit="PoundSterling"),
                "f-one": FacetMetadata(provenance_id=self.SERVED, unit="Yen"),
            },
        )

    @pytest.mark.parametrize(
        ("pool", "probed"),
        [
            pytest.param([TWO_ROLE], [], id="all-two-role"),
            pytest.param(
                [TWO_ROLE, ONE_ROLE, SUBJECT], [[ONE_ROLE, SUBJECT]], id="mixed-sends-the-rest"
            ),
        ],
    )
    def test_the_graph_probe_reads_only_candidates_coverage_cannot_answer(self, pool, probed):
        graph = _ProbeRecordingGraph(self._graph())

        derive_shapes(
            confirmed_svs=pool, graph=graph, place_dcids=[self.KEN], coverage=self._coverage()
        )

        assert graph.probed == probed

    def test_a_two_role_variables_availability_and_units_match_the_graph_probe(self):
        """Under a source scope admitting one of its facets, the coverage answer still lists
        both units the graph lists at the donor place, and no recipient-only facet."""
        with bound_catalogue(FakeCatalogue().scoped(frozenset({self.SERVED}))):
            from_graph, from_coverage = (
                {
                    shape.declared_roles: shape
                    for shape in derive_shapes(
                        confirmed_svs=[self.TWO_ROLE],
                        graph=self._graph(),
                        place_dcids=[self.KEN],
                        coverage=coverage,
                    )
                }[self.ROLES]
                for coverage in (None, self._coverage())
            )

        assert from_graph.place_units == frozenset({"USDollar", "Euro"})
        assert (from_coverage.place_available, from_coverage.place_units) == (
            from_graph.place_available,
            from_graph.place_units,
        )
        assert from_coverage.member_coverage is None

    def test_coverage_reads_each_variables_rows_once_for_every_place(self, monkeypatch):
        """Coverage answers every asked place from one grouped read of the variable's rows,
        timed as its own stage of the probe."""
        reads: list[list[str]] = []
        read_rows = CoverageIndex._candidate_member_rows

        def counted(index, *, variables, bound):
            reads.append(list(variables))
            return read_rows(index, variables=variables, bound=bound)

        monkeypatch.setattr(CoverageIndex, "_candidate_member_rows", counted)
        timings = PhaseTimings()

        with collecting(timings):
            (shape,) = derive_shapes(
                confirmed_svs=[self.TWO_ROLE],
                graph=self._graph(),
                place_dcids=[self.KEN, "country/FRA"],
                coverage=self._coverage(),
            )

        assert reads == [[self.TWO_ROLE]]
        assert shape.place_units == frozenset({"USDollar", "Euro", "PoundSterling"})
        assert "shape_probe_coverage" in timings.snapshot()
