"""Per-axis facet-selection algorithm: `probe_facets`' probe-every-bound-dcid
union, and `select_facet`'s per-axis outcomes.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import replace
from functools import reduce
from itertools import product
from typing import Literal

import pytest

import materialise.facets as facets_module
from dc_graph import earliest_date, latest_date
from dc_graph.client import _build_facet, _parse_ordered_facets
from dc_sdmx import SdmxCapExceededError
from dfo_platform.problem import Alternative
from materialise.errors import (
    CollectionProbeCeilingExceededError,
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationError,
    MaterialisationUsageError,
)
from materialise.facets import (
    _MAX_ALTERNATIVES_PER_ISSUE,
    DEPLOYMENT_FACET_POLICY,
    DEPLOYMENT_FACET_POLICY_VERSION,
    Cadence,
    CoverageAdmission,
    FacetBinding,
    FacetElection,
    FacetSelection,
    MemberProbe,
    PlaceAdmission,
    UnitRequest,
    _facet_from_coverage_candidate,
    cadences_by_member,
    declared_default_policies,
    default_scope_of,
    facet_election,
    fetch_role_bindings,
    joint_dispatch_facet_ids,
    joint_matches,
    merge_facets,
    narrow_to_window,
    overlaps_window,
    probe_collection_candidates,
    probe_facets,
    rank_candidates,
    select_facet,
)
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.pipeline import MaterialisationRequest
from materialise.place_scope import DefaultPlaceScope, DefaultPresence
from materialise.recipes import NonCurrencyUnitError
from materialise.window import RequestedWindow
from series_coverage.facets import facet_candidates
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._group_support import served_alone
from ._scope_availability import NO_SCOPE_READ
from .conftest import FakeGraphReader

_STAT_VAR = "dcid:MalariaGrantDisbursements"

_FACET_KEN = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=402,
    earliest_date="1991",
    latest_date="2024",
)
_FACET_TGO = Facet(
    facet_id="facet2",
    provenance_id="IMF_WEO",
    measurement_method="Amount",
    unit="EUR",
    obs_count=10,
    earliest_date="2005",
    latest_date="2023",
)
_FACET_TGO_DUPLICATE_ID = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=999,
    earliest_date="1991",
    latest_date="2024",
)


def test_probe_facets_unions_across_entities_merging_a_recurring_facet_id():
    # facet1 is reported for both KEN and TGO, merged into one candidate
    # (obs_count summed, dates widened), so ranking on obs_count reflects
    # the whole probed entity set, not `bound_entities`' own order.
    graph = FakeGraphReader(
        facets={
            (_STAT_VAR, "country/KEN"): [_FACET_KEN, _FACET_TGO_DUPLICATE_ID],
            (_STAT_VAR, "country/TGO"): [_FACET_TGO],
        }
    )

    probe = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var=_STAT_VAR,
        bound_entities=["country/KEN", "country/TGO"],
        settle=True,
        graph=graph,
    )

    assert probe.candidates == (
        # Both places date the facet by year, so the merge keeps its annual evidence.
        replace(_FACET_KEN, obs_count=402 + 999, annual_evidence=True),
        _FACET_TGO,
    )
    # Each facet id keeps the places it was admitted at, which the merge drops.
    assert probe.admission == PlaceAdmission(
        places_by_facet={
            "facet1": frozenset({"country/KEN"}),
            "facet2": frozenset({"country/TGO"}),
        }
    )


def test_probe_facets_merge_widens_earliest_and_latest_date():
    early = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=5,
        earliest_date="1990",
        latest_date="2000",
    )
    late = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=7,
        earliest_date="1995",
        latest_date="2010",
    )
    graph = FakeGraphReader(
        facets={
            (_STAT_VAR, "country/KEN"): [early],
            (_STAT_VAR, "country/TGO"): [late],
        }
    )

    (merged,) = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var=_STAT_VAR,
        bound_entities=["country/KEN", "country/TGO"],
        settle=True,
        graph=graph,
    ).candidates

    assert merged.obs_count == 12
    assert merged.earliest_date == "1990"
    assert merged.latest_date == "2010"


_EARTH_COUNTRIES = {("Earth", "Country"): ["country/KEN", "country/TGO", "country/UGA"]}


def test_probe_facets_with_no_bound_entity_merges_every_place_availability_names():
    world = replace(_FACET_KEN, facet_id="facetWorld", obs_count=900)
    graph = FakeGraphReader(
        facets={
            (_STAT_VAR, "country/KEN"): [_FACET_KEN],
            (_STAT_VAR, "country/TGO"): [_FACET_TGO_DUPLICATE_ID, _FACET_TGO],
            (_STAT_VAR, "Earth"): [world],
        },
        child_places=_EARTH_COUNTRIES,
    )
    availability = graph.scope_availability()

    probe = probe_facets(
        stat_var=_STAT_VAR, bound_entities=(), settle=True, graph=graph, availability=availability
    )

    # Kenya's and Togo's rows merge by facet id, and Earth's total is a candidate of its own.
    assert [(c.facet_id, c.obs_count) for c in probe.candidates] == [
        ("facet1", _FACET_KEN.obs_count + _FACET_TGO_DUPLICATE_ID.obs_count),
        ("facet2", _FACET_TGO.obs_count),
        ("facetWorld", 900),
    ]
    # Each facet id is admitted at the places that hold it. Uganda holds no data, so it is
    # neither probed nor fetched.
    assert probe.admission == PlaceAdmission(
        places_by_facet={
            "facet1": frozenset({"country/KEN", "country/TGO"}),
            "facet2": frozenset({"country/TGO"}),
            "facetWorld": frozenset({"Earth"}),
        }
    )
    assert probe.scope == DefaultPlaceScope(
        places=("country/KEN", "country/TGO", "Earth"), every_place=True
    )
    # One availability read names the places. The graph reads no containment and no presence.
    assert availability.reads == [(_STAT_VAR,)]
    assert graph.calls["child_places_batch"] == 0
    assert graph.presence_reads == []
    assert graph.calls["variable_facets_batch"] == 1


def test_probe_facets_past_the_availability_cap_probes_the_countries_with_data():
    world = replace(_FACET_KEN, facet_id="facetWorld")
    graph = FakeGraphReader(
        facets={(_STAT_VAR, "country/KEN"): [_FACET_KEN], (_STAT_VAR, "Earth"): [world]},
        child_places=_EARTH_COUNTRIES,
    )

    probe = probe_facets(
        stat_var=_STAT_VAR,
        bound_entities=(),
        settle=True,
        graph=graph,
        availability=_CappedAvailability(),
    )

    # The countries fallback leaves Earth's total out of a scope that holds a country, and the
    # probe says its places are not every place with data.
    assert probe.scope == DefaultPlaceScope(places=("country/KEN",), every_place=False)
    assert probe.candidates == (_FACET_KEN,)
    assert graph.presence_reads == [
        ((_STAT_VAR,), ("country/KEN", "country/TGO", "country/UGA", "Earth"))
    ]


class _CappedAvailability:
    """Availability whose every read is past `place_scope.MAX_EVERY_PLACE`."""

    def source_key(self) -> str:
        return "fake-availability"

    def availability_values(self, *, variables, component, max_values: int, **_kwargs):
        raise SdmxCapExceededError(
            "too many places", cap_kind="values", cap=max_values, observed=max_values + 1
        )


def test_probe_facets_with_a_worldwide_total_alone_probes_earth():
    world = replace(_FACET_KEN, facet_id="facetWorld")
    graph = FakeGraphReader(facets={(_STAT_VAR, "Earth"): [world]}, child_places=_EARTH_COUNTRIES)

    probe = probe_facets(
        stat_var=_STAT_VAR,
        bound_entities=(),
        settle=True,
        graph=graph,
        availability=graph.scope_availability(),
    )

    assert probe == _admitted(world, "Earth", places=("Earth",))


def test_probe_facets_with_no_bound_entity_reads_no_facet_when_no_place_holds_data():
    graph = FakeGraphReader(child_places=_EARTH_COUNTRIES)

    probe = probe_facets(
        stat_var=_STAT_VAR,
        bound_entities=(),
        settle=True,
        graph=graph,
        availability=graph.scope_availability(),
    )

    assert probe == _NOTHING_ADMITTED
    assert graph.calls["variable_facets_batch"] == 0


def test_a_group_past_the_probe_pair_cap_refuses_before_any_facet_read(monkeypatch):
    # The pair cap stays the backstop for many StatVars at a full scope: 40 at 5,000 places.
    monkeypatch.setattr(facets_module, "_MAX_COLLECTION_PROBE_PAIRS", 5)
    graph = FakeGraphReader(
        facets={
            (stat_var, place): [_FACET_KEN]
            for stat_var in ("A", "B")
            for place in ("country/KEN", "country/TGO", "Earth")
        }
    )

    with pytest.raises(CollectionProbeCeilingExceededError, match="6 stat_var") as refused:
        probe_collection_candidates(
            [_unbound("A"), _unbound("B")],
            settle=True,
            graph=graph,
            availability=graph.scope_availability(),
        )

    assert refused.value.code == "collection_probe_ceiling"
    assert graph.calls["observation_facets_batch_many"] == 0


def test_the_pair_cap_counts_each_member_at_its_own_scope(monkeypatch):
    """Three members over disjoint places: 1 + 1 + 2 = 4 pairs at their own scopes, where
    every member at the union of four places would count 12."""
    monkeypatch.setattr(facets_module, "_MAX_COLLECTION_PROBE_PAIRS", 4)
    graph = FakeGraphReader(
        facets={
            ("A", "country/KEN"): [_FACET_KEN],
            ("B", "country/TGO"): [_FACET_KEN],
            ("C", "country/UGA"): [_FACET_KEN],
            ("C", "Earth"): [_FACET_KEN],
        }
    )
    requests = [_unbound("A"), _unbound("B"), _unbound("C")]

    probed = probe_collection_candidates(
        requests, settle=True, graph=graph, availability=graph.scope_availability()
    )

    assert [member.places for member in probed] == [
        ("country/KEN",),
        ("country/TGO",),
        ("country/UGA", "Earth"),
    ]
    monkeypatch.setattr(facets_module, "_MAX_COLLECTION_PROBE_PAIRS", 3)
    with pytest.raises(CollectionProbeCeilingExceededError, match="4 stat_var"):
        probe_collection_candidates(
            requests, settle=True, graph=graph, availability=graph.scope_availability()
        )


_NOTHING_ADMITTED = MemberProbe(candidates=(), admission=PlaceAdmission(places_by_facet={}))


def _admitted(facet: Facet, *admitted_at: str, places: tuple[str, ...]) -> MemberProbe:
    """A member probe holding `facet` alone, admitted at `admitted_at`, and fetched at
    `places`."""
    return MemberProbe(
        candidates=(facet,),
        admission=PlaceAdmission(places_by_facet={facet.facet_id: frozenset(admitted_at)}),
        places=places,
    )


def _unbound(bare_stat_var: str) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{bare_stat_var}",
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


_BARE = "MalariaGrantDisbursements"


class _ProbeRecordingGraph(FakeGraphReader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.probed: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def observation_facets_batch_many(self, *, stat_vars, entities, deadline=None):
        self.probed.append((tuple(stat_vars), tuple(entities)))
        return super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_members_probed_together_are_each_probed_and_admitted_at_their_own_places(order):
    other, togo, twin = "OtherStatVar", "TogoStatVar", "TwinStatVar"
    world = replace(_FACET_KEN, facet_id="facetWorld")
    graph = _ProbeRecordingGraph(
        facets={
            (_BARE, "country/KEN"): [_FACET_KEN],
            (_BARE, "Earth"): [world],
            (other, "Earth"): [world],
            (togo, "country/TGO"): [_FACET_TGO],
            (twin, "country/TGO"): [_FACET_TGO],
        }
    )
    stat_vars = [*((_BARE, other, togo)[position] for position in order), twin]

    probed = dict(
        zip(
            stat_vars,
            probe_collection_candidates(
                [_unbound(stat_var) for stat_var in stat_vars],
                settle=True,
                graph=graph,
                availability=graph.scope_availability(),
            ),
            strict=True,
        )
    )

    # One batch per distinct place list, each member at its own places. The two members with
    # the same places share one batch.
    assert sorted(graph.probed) == sorted(
        [
            ((_BARE,), ("country/KEN", "Earth")),
            ((other,), ("Earth",)),
            (tuple(sv for sv in stat_vars if sv in (togo, twin)), ("country/TGO",)),
        ]
    )
    assert probed == {
        _BARE: MemberProbe(
            candidates=(_FACET_KEN, world),
            admission=PlaceAdmission(
                places_by_facet={
                    "facet1": frozenset({"country/KEN"}),
                    "facetWorld": frozenset({"Earth"}),
                }
            ),
            places=("country/KEN", "Earth"),
        ),
        togo: _admitted(_FACET_TGO, "country/TGO", places=("country/TGO",)),
        twin: _admitted(_FACET_TGO, "country/TGO", places=("country/TGO",)),
        other: _admitted(world, "Earth", places=("Earth",)),
    }
    # Alone, each gets the same places, candidates and admission.
    for stat_var in stat_vars:
        assert probed[stat_var] == probe_facets(
            stat_var=stat_var,
            bound_entities=(),
            settle=True,
            graph=graph,
            availability=graph.scope_availability(),
        )


def test_the_mint_orders_a_groups_scope_as_search_orders_a_union():
    """A member at Earth alone and a member in Kenya and Togo: Search's union and the mint's
    group scope state the same places in the same order, places by dcid, then Earth, and both
    are every place only when each member's scope is."""
    presence = DefaultPresence(
        countries=(),
        scopes={
            "W": DefaultPlaceScope(places=("Earth",), every_place=True),
            "A": DefaultPlaceScope(places=("country/TGO", "country/KEN"), every_place=False),
        },
    )
    members = [
        MemberProbe(
            candidates=(),
            admission=PlaceAdmission(places_by_facet={}),
            places=presence.scope(stat_var).places,
            every_place=presence.scope(stat_var).every_place,
        )
        for stat_var in ("W", "A")
    ]

    assert default_scope_of(members) == presence.union_scope(["W", "A"])
    assert default_scope_of(members) == DefaultPlaceScope(
        places=("country/KEN", "country/TGO", "Earth"), every_place=False
    )
    assert default_scope_of(members[:1]).every_place


def test_fetch_role_bindings_names_the_default_places_only_for_a_request_binding_no_entity():
    places = ("country/KEN", "country/TGO")

    assert fetch_role_bindings({}, ("observationAbout",), places=places) == {
        "observationAbout": places
    }
    bound = {"observationAbout": ["country/UGA"]}
    assert fetch_role_bindings(bound, ("observationAbout",), places=places) is bound
    # A role-bearing request keeps its empty bindings. Coverage supplies its filter.
    assert fetch_role_bindings({}, ("recipient",), places=places) == {}


def test_probe_facets_role_bearing_with_no_bound_role_covers_every_coverage_row():
    coverage = build_coverage_index(
        [
            ("MalariaGrantDisbursements", "facet1", "country/FRA", "country/KEN", [2019, 2020]),
            ("MalariaGrantDisbursements", "facet1", "country/DEU", "country/TGO", [2020]),
        ],
        role_names={
            "MalariaGrantDisbursements": RoleNames(
                donor="financingProvider", recipient="financingRecipient"
            )
        },
    )
    graph = FakeGraphReader()

    unbound = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var="MalariaGrantDisbursements",
        bound_entities=(),
        settle=True,
        graph=graph,
        declared_roles=("financingProvider", "financingRecipient"),
        role_bindings={},
        coverage=coverage,
    )
    kenya = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var="MalariaGrantDisbursements",
        bound_entities=["country/KEN"],
        settle=True,
        graph=graph,
        declared_roles=("financingProvider", "financingRecipient"),
        role_bindings={"financingRecipient": ["country/KEN"]},
        coverage=coverage,
    )

    assert [(c.facet_id, c.obs_count) for c in unbound.candidates] == [("facet1", 3)]
    assert [(c.facet_id, c.obs_count) for c in kenya.candidates] == [("facet1", 2)]
    # Coverage rows serve unmasked, with a role bound or with none.
    assert unbound.admission == kenya.admission == CoverageAdmission()
    assert sum(graph.calls.values()) == 0


def test_probe_facets_forwards_deadline_to_the_graph_call():
    graph = FakeGraphReader(facets={(_STAT_VAR, "country/KEN"): [_FACET_KEN]})
    deadlines: list[float | None] = []
    original = graph.variable_facets_batch

    def recording_variable_facets_batch(**kwargs):
        deadlines.append(kwargs.get("deadline"))
        return original(**kwargs)

    graph.variable_facets_batch = recording_variable_facets_batch  # type: ignore[method-assign]
    deadline = time.monotonic() + 30

    probe_facets(
        availability=NO_SCOPE_READ,
        stat_var=_STAT_VAR,
        bound_entities=["country/KEN"],
        settle=True,
        graph=graph,
        deadline=deadline,
    )

    assert deadlines == [deadline]


def test_probe_facets_observation_about_declared_roles_takes_graph_path_byte_identical():
    # The default `declared_roles` and an explicit `("observationAbout",)`
    # both take the graph path, ignoring `role_bindings` entirely, even
    # when a caller supplies them: the observationAbout case is unaffected
    # by the role-aware branch this module gained.
    graph = FakeGraphReader(facets={(_STAT_VAR, "country/KEN"): [_FACET_KEN]})

    via_default = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var=_STAT_VAR,
        bound_entities=["country/KEN"],
        settle=True,
        graph=graph,
    )
    via_explicit = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var=_STAT_VAR,
        bound_entities=["country/KEN"],
        settle=True,
        graph=graph,
        declared_roles=("observationAbout",),
        role_bindings={"observationAbout": ["country/KEN"]},
    )

    assert via_default == via_explicit
    assert via_default.candidates == (_FACET_KEN,)
    assert graph.calls["variable_facets_batch"] == 2


def test_probe_facets_role_bearing_no_coverage_raises_usage_error():
    # Coverage is the source of role-bearing candidates; omission is a caller error.
    with pytest.raises(MaterialisationUsageError, match="MalariaGrantDisbursements"):
        probe_facets(
            availability=NO_SCOPE_READ,
            stat_var="MalariaGrantDisbursements",
            bound_entities=["country/KEN"],
            settle=True,
            graph=FakeGraphReader(),
            declared_roles=("financingRecipient",),
            role_bindings={"financingRecipient": ["country/KEN"]},
        )


def _coverage_naming_another_stat_var():
    return build_coverage_index(
        [("OtherStatVar", "facet1", "country/FRA", "country/KEN", [2020])],
        role_names={
            "OtherStatVar": RoleNames(donor="financingProvider", recipient="financingRecipient")
        },
    )


def test_probe_facets_two_role_uncovered_stat_var_returns_no_candidates():
    # An uncovered two-role StatVar has no pair series, so it returns no candidates and the
    # graph is not read.
    graph = FakeGraphReader(facets={("MalariaGrantDisbursements", "country/KEN"): [_FACET_KEN]})

    probe = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var="MalariaGrantDisbursements",
        bound_entities=["country/KEN"],
        settle=True,
        graph=graph,
        declared_roles=("financingProvider", "financingRecipient"),
        role_bindings={"financingRecipient": ["country/KEN"]},
        coverage=_coverage_naming_another_stat_var(),
    )

    assert probe == MemberProbe(candidates=(), admission=CoverageAdmission())
    assert graph.calls["variable_facets_batch"] == 0


def test_probe_facets_one_role_uncovered_stat_var_returns_no_candidates():
    # A one-role StatVar reads coverage as a two-role one does: the index holds its rows
    # with no donor, so one it does not name has no candidates and the graph is not read.
    graph = FakeGraphReader(facets={("DebtService", "country/GHA"): [_FACET_KEN]})

    probe = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var="DebtService",
        bound_entities=["country/GHA"],
        settle=True,
        graph=graph,
        declared_roles=("financingRecipient",),
        role_bindings={"financingRecipient": ["country/GHA"]},
        coverage=_coverage_naming_another_stat_var(),
    )

    assert probe == MemberProbe(candidates=(), admission=CoverageAdmission())
    assert graph.calls["variable_facets_batch"] == 0


def test_probe_facets_coverage_candidates_carry_both_year_bounds():
    # Each coverage facet's first and last year become its date bounds, so a facet
    # that starts after the window's end no longer counts as overlapping it.
    coverage = build_coverage_index(
        [
            ("MalariaGrantDisbursements", "fRecorded", "", "country/KEN", [2020, 2021, 2022]),
            ("MalariaGrantDisbursements", "fEstimated", "", "country/KEN", [2025, 2026]),
        ],
        role_names={
            "MalariaGrantDisbursements": RoleNames(donor=None, recipient="financingRecipient")
        },
    )

    candidates = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var="MalariaGrantDisbursements",
        bound_entities=["country/KEN"],
        settle=True,
        graph=FakeGraphReader(),
        declared_roles=("financingRecipient",),
        role_bindings={"financingRecipient": ["country/KEN"]},
        coverage=coverage,
    ).candidates

    bounds = {c.facet_id: (c.earliest_date, c.latest_date) for c in candidates}
    assert bounds == {"fRecorded": ("2020", "2022"), "fEstimated": ("2025", "2026")}
    [kept] = narrow_to_window(candidates, window=_range(2021, 2022))
    assert kept.facet_id == "fRecorded"


def test_probe_facets_role_bearing_without_role_bindings_raises_usage_error():
    with pytest.raises(MaterialisationUsageError, match="role_bindings"):
        probe_facets(
            availability=NO_SCOPE_READ,
            stat_var="MalariaGrantDisbursements",
            bound_entities=["country/KEN"],
            settle=True,
            graph=FakeGraphReader(),
            declared_roles=("financingRecipient",),
        )


def test_bound_matching_candidate_found():
    graph = FakeGraphReader()

    selection = select_facet(
        "unit",
        FacetBinding(state="bound", ref="USD"),
        candidates=[_FACET_KEN, _FACET_TGO],
        graph=graph,
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": FacetBinding(state="bound", ref="USD")}, graph=graph, stat_var=_STAT_VAR
        ),
    )

    assert selection.basis == "bound"
    assert selection.ref == "USD"
    assert selection.dispatch_facet_id == "facet1"
    assert selection.default_policy is None
    assert selection.default_policy_version is None
    assert selection.selection_reason is None


def test_bound_null_ref_selects_the_facet_that_declares_no_value():
    """A null bound ref matches the candidate without a measurement method, and a bound
    method ref never matches it."""
    recorded = replace(_FACET_KEN, facet_id="recorded", measurement_method=None)
    estimated = replace(_FACET_KEN, facet_id="estimated", measurement_method="Estimated")
    candidates = [estimated, recorded]

    absent = select_facet(
        "measurementMethod",
        FacetBinding(state="bound", ref=None),
        candidates=candidates,
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"measurementMethod": FacetBinding(state="bound", ref=None)},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )
    named = select_facet(
        "measurementMethod",
        FacetBinding(state="bound", ref="Estimated"),
        candidates=candidates,
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"measurementMethod": FacetBinding(state="bound", ref="Estimated")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )

    assert (absent.basis, absent.ref, absent.dispatch_facet_id) == ("bound", None, "recorded")
    assert (named.ref, named.dispatch_facet_id) == ("Estimated", "estimated")


def test_a_bound_null_ref_with_no_match_names_no_declared_value():
    """A null bound ref no candidate matches is refused in the client's terms: it asks for
    the facets with no declared value, never for a value named `None`."""
    estimated = replace(_FACET_KEN, facet_id="estimated", measurement_method="Estimated")

    with pytest.raises(FacetNotAvailableError) as excinfo:
        select_facet(
            "measurementMethod",
            FacetBinding(state="bound", ref=None),
            candidates=[estimated],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"measurementMethod": FacetBinding(state="bound", ref=None)},
                graph=FakeGraphReader(),
                stat_var=_STAT_VAR,
            ),
        )

    assert "bound to no declared value" in str(excinfo.value)
    assert "None" not in str(excinfo.value)


def test_bound_no_matching_candidate_raises_facet_not_available():
    graph = FakeGraphReader(labels={"USD": "US Dollar"})

    with pytest.raises(MaterialisationError) as excinfo:
        select_facet(
            "unit",
            FacetBinding(state="bound", ref="GBP"),
            candidates=[_FACET_KEN, _FACET_TGO],
            graph=graph,
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"unit": FacetBinding(state="bound", ref="GBP")}, graph=graph, stat_var=_STAT_VAR
            ),
        )

    assert isinstance(excinfo.value, FacetNotAvailableError)
    assert excinfo.value.code == "facet_not_available"
    assert excinfo.value.alternatives is not None
    assert Alternative(ref="USD", label="US Dollar") in excinfo.value.alternatives
    # EUR has no fake label registered, so it falls back to the bare ref.
    assert Alternative(ref="EUR", label="EUR") in excinfo.value.alternatives
    assert excinfo.value.alternatives_truncated is False


def _unit_facet(facet_id: str | None, unit: str) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit=unit,
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )


def test_alternatives_caps_node_label_lookups_at_first_seen_distinct_values():
    candidates = [
        _unit_facet(f"facet{i}", f"UNIT{i}") for i in range(_MAX_ALTERNATIVES_PER_ISSUE + 5)
    ]
    graph = FakeGraphReader()

    with pytest.raises(MaterialisationError) as excinfo:
        select_facet(
            "unit",
            FacetBinding(state="bound", ref="NOMATCH"),
            candidates=candidates,
            graph=graph,
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"unit": FacetBinding(state="bound", ref="NOMATCH")},
                graph=graph,
                stat_var=_STAT_VAR,
            ),
        )

    assert isinstance(excinfo.value, FacetNotAvailableError)
    assert excinfo.value.alternatives is not None
    assert graph.calls["node_label"] == _MAX_ALTERNATIVES_PER_ISSUE
    assert [alt.ref for alt in excinfo.value.alternatives] == [
        f"UNIT{i}" for i in range(_MAX_ALTERNATIVES_PER_ISSUE)
    ]
    assert excinfo.value.alternatives_truncated is True


def test_alternatives_excludes_a_facet_id_none_candidate_which_does_not_count_toward_the_cap():
    # If `facet_id=None` entered alternatives, it would displace the last dispatchable value.
    dispatchable = [
        _unit_facet(f"facet{i}", f"UNIT{i}") for i in range(_MAX_ALTERNATIVES_PER_ISSUE)
    ]
    candidates = [_unit_facet(None, "UNDISPATCHABLE"), *dispatchable]
    graph = FakeGraphReader()

    with pytest.raises(MaterialisationError) as excinfo:
        select_facet(
            "unit",
            FacetBinding(state="bound", ref="NOMATCH"),
            candidates=candidates,
            graph=graph,
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"unit": FacetBinding(state="bound", ref="NOMATCH")},
                graph=graph,
                stat_var=_STAT_VAR,
            ),
        )

    assert isinstance(excinfo.value, FacetNotAvailableError)
    assert excinfo.value.alternatives is not None
    alt_refs = [alt.ref for alt in excinfo.value.alternatives]
    assert "UNDISPATCHABLE" not in alt_refs
    assert alt_refs == [f"UNIT{i}" for i in range(_MAX_ALTERNATIVES_PER_ISSUE)]
    assert graph.calls["node_label"] == _MAX_ALTERNATIVES_PER_ISSUE
    # Exactly `_MAX_ALTERNATIVES_PER_ISSUE` values are present.
    # The alternatives list is untruncated.
    assert excinfo.value.alternatives_truncated is False


def test_unbound_no_policy_no_candidates_raises_default_facet_undefined():
    # No graph-declared policy and no dispatchable candidate at all. The
    # deployment default policy has nothing to rank, so this refuses.
    graph = FakeGraphReader()

    with pytest.raises(MaterialisationError) as excinfo:
        select_facet(
            "unit",
            FacetBinding(state="unbound"),
            candidates=[],
            graph=graph,
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"unit": FacetBinding(state="unbound")}, graph=graph, stat_var=_STAT_VAR
            ),
        )

    assert isinstance(excinfo.value, DefaultFacetUndefinedError)
    assert excinfo.value.code == "default_facet_undefined"


def test_unbound_no_policy_sole_candidate_binds_via_deployment_default():
    graph = FakeGraphReader()

    selection = select_facet(
        "unit",
        FacetBinding(state="unbound"),
        candidates=[_FACET_KEN],
        graph=graph,
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": FacetBinding(state="unbound")}, graph=graph, stat_var=_STAT_VAR
        ),
    )

    assert selection.basis == "deployment_fallback"
    assert selection.ref == "USD"
    assert selection.dispatch_facet_id == "facet1"
    assert selection.default_policy == DEPLOYMENT_FACET_POLICY
    assert selection.default_policy_version == DEPLOYMENT_FACET_POLICY_VERSION
    assert selection.selection_reason == "sole_candidate"


def test_unbound_no_policy_multiple_agreeing_candidates_binds_most_observations():
    # Multiple candidates agree on provenance, so the reason is
    # `most_observations`.
    agreeing = replace(_FACET_TGO, facet_id="facet3", provenance_id="OECD_DAC2A", obs_count=5)
    graph = FakeGraphReader()

    selection = select_facet(
        "provenance",
        FacetBinding(state="unbound"),
        candidates=[_FACET_KEN, agreeing],
        graph=graph,
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"provenance": FacetBinding(state="unbound")}, graph=graph, stat_var=_STAT_VAR
        ),
    )

    assert selection.ref == "OECD_DAC2A"
    assert selection.basis == "deployment_fallback"
    assert selection.selection_reason == "most_observations"


def test_unbound_no_policy_disagreement_binds_top_ranked_candidates_value():
    # KEN (402 obs) outranks TGO (10 obs) on obs_count. The deployment
    # default picks KEN's own provenance even though TGO disagrees.
    selection = select_facet(
        "provenance",
        FacetBinding(state="unbound"),
        candidates=[_FACET_KEN, _FACET_TGO],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"provenance": FacetBinding(state="unbound")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )

    assert selection.ref == "OECD_DAC2A"
    assert selection.basis == "deployment_fallback"
    assert selection.selection_reason == "most_observations_over_disagreement"
    assert selection.dispatch_facet_id == "facet1"
    # The trace names both values in rank order, with the figures the ranking read; no window
    # and no breadth leave `years` and `places` unread.
    assert selection.choice_entry() == {
        "axis": "provenance",
        "candidates": [
            {"ref": "OECD_DAC2A", "places": None, "years": None, "observations": 402},
            {"ref": "IMF_WEO", "places": None, "years": None, "observations": 10},
        ],
    }


def test_unbound_no_policy_axis_absent_on_top_ranked_facet_binds_null_not_a_refusal():
    top_ranked_absent = replace(_FACET_KEN, measurement_method=None)
    selection = select_facet(
        "measurementMethod",
        FacetBinding(state="unbound"),
        candidates=[top_ranked_absent, _FACET_TGO],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"measurementMethod": FacetBinding(state="unbound")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )

    assert selection.ref is None
    assert selection.basis == "deployment_fallback"
    assert selection.selection_reason == "absent_on_selected_facet"
    assert selection.dispatch_facet_id == "facet1"


def test_unbound_no_policy_provenance_absent_on_top_ranked_facet_raises_default_undefined():
    # Citation assembly requires a concrete provenance ref, so an absent
    # provenance raises `DefaultFacetUndefinedError`.
    top_ranked_absent = replace(_FACET_KEN, provenance_id=None)

    with pytest.raises(DefaultFacetUndefinedError):
        select_facet(
            "provenance",
            FacetBinding(state="unbound"),
            candidates=[top_ranked_absent, _FACET_TGO],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"provenance": FacetBinding(state="unbound")},
                graph=FakeGraphReader(),
                stat_var=_STAT_VAR,
            ),
        )


def test_unbound_no_policy_narrows_by_a_sibling_bound_axis():
    # `provenance` is bound to TGO's own value, which only TGO's candidate
    # carries, narrowing `unit`'s own default-policy pool down to just
    # TGO's candidate, even though KEN's has far more observations.
    all_bindings = {"provenance": FacetBinding(state="bound", ref="IMF_WEO")}

    selection = select_facet(
        "unit",
        FacetBinding(state="unbound"),
        candidates=[_FACET_KEN, _FACET_TGO],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        all_bindings=all_bindings,
        window=None,
        policies=declared_default_policies(
            all_bindings, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert selection.ref == "EUR"
    assert selection.selection_reason == "sole_candidate"


def test_unbound_no_policy_narrowing_to_nothing_raises_default_facet_undefined():
    all_bindings = {"provenance": FacetBinding(state="bound", ref="NO_SUCH_PROVENANCE")}

    with pytest.raises(MaterialisationError) as excinfo:
        select_facet(
            "unit",
            FacetBinding(state="unbound"),
            candidates=[_FACET_KEN, _FACET_TGO],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            all_bindings=all_bindings,
            window=None,
            policies=declared_default_policies(
                all_bindings, graph=FakeGraphReader(), stat_var=_STAT_VAR
            ),
        )

    assert isinstance(excinfo.value, DefaultFacetUndefinedError)


def test_rank_candidates_orders_by_obs_count_then_recency_then_lexicographic_then_facet_id():
    most_observed = Facet(
        facet_id="fZ",
        provenance_id="Z",
        measurement_method=None,
        unit=None,
        obs_count=100,
        earliest_date="2000",
        latest_date="2010",
    )
    recent_tiebreak_winner = Facet(
        facet_id="fB",
        provenance_id="B",
        measurement_method=None,
        unit=None,
        obs_count=10,
        earliest_date="2000",
        latest_date="2023",
    )
    recent_tiebreak_loser = Facet(
        facet_id="fA",
        provenance_id="A",
        measurement_method=None,
        unit=None,
        obs_count=10,
        earliest_date="2000",
        latest_date="2020",
    )
    facet_id_tiebreak_winner = Facet(
        facet_id="fA",
        provenance_id="same",
        measurement_method="same",
        unit="same",
        obs_count=1,
        earliest_date=None,
        latest_date=None,
    )
    facet_id_tiebreak_loser = Facet(
        facet_id="fB",
        provenance_id="same",
        measurement_method="same",
        unit="same",
        obs_count=1,
        earliest_date=None,
        latest_date=None,
    )

    ranked = rank_candidates(
        [
            recent_tiebreak_loser,
            most_observed,
            facet_id_tiebreak_loser,
            recent_tiebreak_winner,
            facet_id_tiebreak_winner,
        ],
        window=None,
    )

    assert ranked[0] is most_observed
    assert ranked[1] is recent_tiebreak_winner
    assert ranked[2] is recent_tiebreak_loser
    assert ranked[3] is facet_id_tiebreak_winner
    assert ranked[4] is facet_id_tiebreak_loser


# One StatVar and entity with recorded values (no method) and projections in their own
# `Estimated` facet, the IDS layout. The recorded facet has more observations.
_RECORDED = Facet(
    facet_id="facetRecorded",
    provenance_id="WorldBankIDS",
    measurement_method=None,
    unit="USDollar",
    obs_count=55,
    earliest_date="1970",
    latest_date="2024",
)
_ESTIMATED = Facet(
    facet_id="facetEstimated",
    provenance_id="WorldBankIDS",
    measurement_method="Estimated",
    unit="USDollar",
    obs_count=8,
    earliest_date="2025",
    latest_date="2032",
)


def _range(start_year: int | None, end_year: int | None) -> RequestedWindow:
    return RequestedWindow(kind="range", start_year=start_year, end_year=end_year)


@pytest.mark.parametrize(
    ("window", "expected_method", "expected_facet_id"),
    [
        (None, None, "facetRecorded"),
        (RequestedWindow(kind="all"), None, "facetRecorded"),
        (RequestedWindow(kind="latest"), None, "facetRecorded"),
        (_range(2027, 2027), "Estimated", "facetEstimated"),
        (_range(2027, None), "Estimated", "facetEstimated"),
        (_range(2020, 2027), None, "facetRecorded"),
    ],
)
def test_window_selects_the_recorded_or_estimated_facet_by_overlap(
    window, expected_method, expected_facet_id
):
    """With no year range the recorded facet wins on count. A 2027 window reaches only the
    `Estimated` facet. A 2020-2027 window overlaps both, and the recorded facet wins on count."""
    selection = select_facet(
        "measurementMethod",
        FacetBinding(state="unbound"),
        candidates=[_RECORDED, _ESTIMATED],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=window,
        policies=declared_default_policies(
            {"measurementMethod": FacetBinding(state="unbound")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )

    assert selection.ref == expected_method
    assert selection.dispatch_facet_id == expected_facet_id
    assert selection.default_policy == DEPLOYMENT_FACET_POLICY
    assert selection.default_policy_version == DEPLOYMENT_FACET_POLICY_VERSION


def test_window_selection_reasons_describe_the_narrowed_pool():
    """Without a window the two facets disagree on method. A 2027 window leaves one."""
    unwindowed = select_facet(
        "measurementMethod",
        FacetBinding(state="unbound"),
        candidates=[_RECORDED, _ESTIMATED],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"measurementMethod": FacetBinding(state="unbound")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )
    windowed = select_facet(
        "measurementMethod",
        FacetBinding(state="unbound"),
        candidates=[_RECORDED, _ESTIMATED],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=_range(2027, 2027),
        policies=declared_default_policies(
            {"measurementMethod": FacetBinding(state="unbound")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )

    assert unwindowed.selection_reason == "absent_on_selected_facet"
    assert windowed.selection_reason == "sole_candidate"


def test_window_that_no_candidate_reaches_keeps_the_whole_pool():
    """A 1950 window overlaps neither facet, so the lifetime ranking decides."""
    selection = select_facet(
        "measurementMethod",
        FacetBinding(state="unbound"),
        candidates=[_ESTIMATED, _RECORDED],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=_range(1950, 1950),
        policies=declared_default_policies(
            {"measurementMethod": FacetBinding(state="unbound")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )

    assert selection.dispatch_facet_id == "facetRecorded"
    assert narrow_to_window([_ESTIMATED, _RECORDED], window=_range(1950, 1950)) == [
        _ESTIMATED,
        _RECORDED,
    ]


def test_window_bound_axis_ignores_the_window():
    """A bound method serves the recorded-era request from the `Estimated` facet regardless."""
    selection = select_facet(
        "measurementMethod",
        FacetBinding(state="bound", ref="Estimated"),
        candidates=[_RECORDED, _ESTIMATED],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=_range(2000, 2010),
        policies=declared_default_policies(
            {"measurementMethod": FacetBinding(state="bound", ref="Estimated")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )

    assert selection.basis == "bound"
    assert selection.dispatch_facet_id == "facetEstimated"


def test_rank_candidates_puts_overlapping_candidates_first_then_the_usual_order():
    later_small = replace(_ESTIMATED, facet_id="fLaterSmall", obs_count=2)
    ranked = rank_candidates([_RECORDED, later_small, _ESTIMATED], window=_range(2026, None))

    assert ranked == (_ESTIMATED, later_small, _RECORDED)
    assert rank_candidates([_ESTIMATED, _RECORDED], window=None) == (_RECORDED, _ESTIMATED)


_OTHER_SOURCE = replace(_RECORDED, facet_id="fOther", provenance_id="OtherSource", obs_count=80)
"""Another source's recorded facet, covering the same 9 years of a 2016 to 2030 window as
`_RECORDED` with more observations, so it outranks `_RECORDED` alone."""

_RECORDED_LATER = replace(
    _RECORDED, facet_id="fRecordedLater", obs_count=2, earliest_date="2025", latest_date="2026"
)
"""`_RECORDED`'s provenance, method, unit and cadence in another facet, reaching 2026: an
election of either serves both, so their series runs to 2026."""

_ESTIMATED_TO_2025 = replace(_ESTIMATED, obs_count=6, earliest_date="2020", latest_date="2025")
"""Estimates reaching 2025, past `_RECORDED` alone but short of `_RECORDED_LATER`."""

_ESTIMATED_TO_2024 = replace(_ESTIMATED, obs_count=5, earliest_date="2020", latest_date="2024")
"""Estimates reaching 2024, past neither `_RECORDED` nor `_RECORDED_LATER`."""


@pytest.mark.parametrize(
    ("candidates", "window", "expected"),
    [
        # The estimates cover more of the window than the series they continue, and follow
        # it as one candidate.
        pytest.param(
            [_ESTIMATED, _RECORDED], _range(2022, 2030), (_RECORDED, _ESTIMATED), id="joined"
        ),
        # Joined, the recorded series and its estimates span 2016 to 2030 and outrank another
        # source's 2016 to 2024 series.
        pytest.param(
            [_OTHER_SOURCE, _ESTIMATED, _RECORDED],
            _range(2016, 2030),
            (_RECORDED, _ESTIMATED, _OTHER_SOURCE),
            id="joined-span-outranks-another-source",
        ),
        # Estimates a dispatch of the recorded series would not serve rank alone: monthly
        # ones sit at another cadence, and ones that start after 2030 hold no point in it.
        pytest.param(
            [_OTHER_SOURCE, replace(_ESTIMATED, observation_period="P1M"), _RECORDED],
            _range(2016, 2030),
            (_OTHER_SOURCE, _RECORDED, replace(_ESTIMATED, observation_period="P1M")),
            id="estimates-at-another-cadence-rank-alone",
        ),
        pytest.param(
            [_OTHER_SOURCE, replace(_ESTIMATED, earliest_date="2031"), _RECORDED],
            _range(2016, 2030),
            (_OTHER_SOURCE, _RECORDED, replace(_ESTIMATED, earliest_date="2031")),
            id="estimates-past-the-window-rank-alone",
        ),
        # Nothing continues under these windows, so the usual order stands.
        pytest.param(
            [_ESTIMATED, _RECORDED], None, (_RECORDED, _ESTIMATED), id="control-no-window"
        ),
        pytest.param(
            [_ESTIMATED, _RECORDED],
            _range(2026, 2030),
            (_ESTIMATED, _RECORDED),
            id="control-recorded-series-misses-the-window",
        ),
        pytest.param(
            [_OTHER_SOURCE, _RECORDED],
            _range(2016, 2030),
            (_OTHER_SOURCE, _RECORDED),
            id="control-no-estimates",
        ),
        # Each series continues past its own end, as a dispatch serving the selection does:
        # estimates ending 2025 continue `_RECORDED`, which ends in 2024, though
        # `_RECORDED_LATER` reaches 2026, and joined they lead.
        pytest.param(
            [_OTHER_SOURCE, _RECORDED, _RECORDED_LATER, _ESTIMATED_TO_2025],
            _range(2016, 2030),
            (_RECORDED, _ESTIMATED_TO_2025, _OTHER_SOURCE, _RECORDED_LATER),
            id="estimates-past-one-series-of-the-selection-join",
        ),
        # Estimates ending 2024 reach past no series of the selection, so rank alone and the
        # other source leads.
        pytest.param(
            [_OTHER_SOURCE, _RECORDED, _RECORDED_LATER, _ESTIMATED_TO_2024],
            _range(2016, 2030),
            (_OTHER_SOURCE, _RECORDED, _ESTIMATED_TO_2024, _RECORDED_LATER),
            id="estimates-short-of-every-series-of-the-selection-rank-alone",
        ),
        pytest.param(
            [_OTHER_SOURCE, _RECORDED, _ESTIMATED_TO_2025],
            _range(2016, 2030),
            (_RECORDED, _ESTIMATED_TO_2025, _OTHER_SOURCE),
            id="control-estimates-past-the-selections-last-year-join",
        ),
        pytest.param(
            [replace(_ESTIMATED, unit="Euro", obs_count=80), _RECORDED],
            _range(2022, 2030),
            (replace(_ESTIMATED, unit="Euro", obs_count=80), _RECORDED),
            id="control-estimates-in-another-unit-rank-alone",
        ),
        # On a full tie, a method classed estimated ranks after no method, which sorts last
        # among unclassed methods.
        pytest.param(
            [replace(_RECORDED, facet_id="fE", measurement_method="Estimated"), _RECORDED],
            None,
            (_RECORDED, replace(_RECORDED, facet_id="fE", measurement_method="Estimated")),
            id="tie-estimated-method-after-no-method",
        ),
        # The same tie across two sources: the estimates rank after the other source's
        # reported series, though their source's name sorts first.
        pytest.param(
            [
                replace(
                    _RECORDED,
                    facet_id="fE",
                    provenance_id="AlphaSource",
                    measurement_method="Estimated",
                ),
                _RECORDED,
            ],
            None,
            (
                _RECORDED,
                replace(
                    _RECORDED,
                    facet_id="fE",
                    provenance_id="AlphaSource",
                    measurement_method="Estimated",
                ),
            ),
            id="tie-estimated-method-after-another-sources-reported-series",
        ),
    ],
)
def test_a_series_and_the_estimates_continuing_it_rank_as_one_candidate(
    candidates, window, expected
):
    assert rank_candidates(candidates, window=window) == expected


@pytest.mark.parametrize(
    ("candidates", "served"),
    [
        pytest.param(
            [_OTHER_SOURCE, _RECORDED, _RECORDED_LATER, _ESTIMATED_TO_2025],
            (_RECORDED, _RECORDED_LATER, _ESTIMATED_TO_2025),
            id="estimates-past-one-series-of-the-selection",
        ),
        pytest.param(
            [_OTHER_SOURCE, _RECORDED, _RECORDED_LATER, _ESTIMATED_TO_2024],
            (_OTHER_SOURCE,),
            id="estimates-short-of-every-series-of-the-selection",
        ),
        pytest.param(
            [_OTHER_SOURCE, _RECORDED, _ESTIMATED_TO_2025],
            (_RECORDED, _ESTIMATED_TO_2025),
            id="control-estimates-past-the-selections-last-year",
        ),
    ],
)
def test_a_dispatch_serves_the_series_and_estimates_the_ranking_credited(candidates, served):
    """The ranking credits a series with the estimates its dispatch serves, so the
    election is never won on estimates the version then leaves out."""
    assert served_alone(candidates, stat_var=_STAT_VAR, window=_range(2016, 2030)) == served


def test_estimates_continuing_two_selections_leave_the_election_to_their_own_series():
    """Two methods of one source, one reported to 2023 and one to 2022, and estimates from
    2024 that continue either. Each selection is credited with the estimates its own
    dispatch would serve, so the series reaching 2023 wins in every candidate order."""
    longer = replace(
        _RECORDED, facet_id="fLonger", obs_count=4, earliest_date="2020", latest_date="2023"
    )
    shorter = replace(
        _RECORDED,
        facet_id="fShorter",
        measurement_method="Standard",
        obs_count=3,
        earliest_date="2020",
        latest_date="2022",
    )
    estimates = replace(_ESTIMATED, obs_count=7, earliest_date="2024", latest_date="2030")
    window = _range(2020, 2030)

    for order in itertools.permutations([longer, shorter, estimates]):
        assert served_alone(order, stat_var=_STAT_VAR, window=window) == (
            longer,
            estimates,
        ), [f.facet_id for f in order]
        assert rank_candidates(order, window=window)[0] == longer


def test_unknown_earliest_date_overlaps_when_latest_date_reaches_the_window_start():
    """A candidate without `earliest_date` is bounded by its `latest_date` alone."""
    reaches = replace(_ESTIMATED, facet_id="fReaches", earliest_date=None, latest_date="2030")
    stops_short = replace(_RECORDED, facet_id="fShort", earliest_date=None, latest_date="2024")

    assert narrow_to_window([stops_short, reaches], window=_range(2027, 2027)) == [reaches]
    assert narrow_to_window([stops_short, reaches], window=_range(2020, 2020)) == [
        stops_short,
        reaches,
    ]


def test_a_dated_candidate_overlaps_a_window_only_where_it_holds_a_point():
    """A facet recorded to 2015 and projected from 2027 spans 2016 to 2026 but holds no point
    there: with all its dates it misses that window, and another facet with points in it
    leads the ranking. Without complete dates the span decides, as before."""
    years = [*range(2010, 2016), *range(2027, 2031)]
    gapped = replace(
        _RECORDED,
        facet_id="fGap",
        obs_count=len(years),
        earliest_date="2010",
        latest_date="2030",
        dates=tuple(str(y) for y in years),
        dates_complete=True,
    )
    inside = replace(
        _RECORDED,
        facet_id="fInside",
        obs_count=2,
        earliest_date="2021",
        latest_date="2022",
        dates=("2021", "2022"),
        dates_complete=True,
    )
    window = _range(2018, 2026)

    assert narrow_to_window([gapped, inside], window=window) == [inside]
    assert rank_candidates([gapped, inside], window=window)[0] == inside
    assert narrow_to_window([gapped, inside], window=_range(2028, None)) == [gapped]
    unsure = replace(gapped, dates_complete=False)
    assert narrow_to_window([unsure, inside], window=window) == [unsure, inside]


def test_a_coverage_candidate_carries_its_years_with_data_as_dates():
    """Coverage rows' year masks give materialise's coverage candidates their points, so a
    role-bearing election narrows by them too."""
    index = build_coverage_index(
        [("V", "f1", "country/D", "country/R", [*range(2010, 2016), *range(2027, 2031)])],
        role_names={"V": RoleNames(donor="creditor", recipient="debtor")},
    )
    (variable,) = index.filter_candidates(variables=["V"], bound={})
    (candidate,) = facet_candidates(variable.facets, metadata={})
    facet = _facet_from_coverage_candidate(candidate)

    assert facet.dates == tuple(str(y) for y in (*range(2010, 2016), *range(2027, 2031)))
    assert not overlaps_window(facet, window=_range(2018, 2026))


def test_undated_candidate_is_never_excluded_by_the_window():
    undated = replace(_RECORDED, facet_id="fUndated", earliest_date=None, latest_date=None)

    assert narrow_to_window([undated, _ESTIMATED], window=_range(2027, 2027)) == [
        undated,
        _ESTIMATED,
    ]


def test_rank_candidates_sorts_a_missing_value_last_on_every_tiebreak_key():
    with_provenance = replace(_FACET_KEN, facet_id="fA", obs_count=1, provenance_id="P")
    without_provenance = replace(_FACET_KEN, facet_id="fB", obs_count=1, provenance_id=None)

    ranked = rank_candidates([without_provenance, with_provenance], window=None)

    assert ranked == (with_provenance, without_provenance)


def test_unbound_policy_resolved_ref_present_among_candidates():
    policy = FacetDefaultPolicy(
        resolved_ref="USD",
        policy_node="policy/unit-default",
        policy_version="v1",
        selection_reason="most-widely-reported currency",
    )
    graph = FakeGraphReader(policies={(_STAT_VAR, "unit"): policy})

    selection = select_facet(
        "unit",
        FacetBinding(state="unbound"),
        candidates=[_FACET_KEN, _FACET_TGO],
        graph=graph,
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": FacetBinding(state="unbound")}, graph=graph, stat_var=_STAT_VAR
        ),
    )

    assert selection.basis == "declared_default"
    assert selection.ref == "USD"
    assert selection.dispatch_facet_id == "facet1"
    assert selection.default_policy == "policy/unit-default"
    assert selection.default_policy_version == "v1"
    assert selection.selection_reason == "most-widely-reported currency"


def test_unbound_policy_resolved_ref_absent_raises_facet_not_available_not_undefined():
    policy = FacetDefaultPolicy(
        resolved_ref="GBP",
        policy_node="policy/unit-default",
        policy_version="v1",
        selection_reason="most-widely-reported currency",
    )
    graph = FakeGraphReader(policies={(_STAT_VAR, "unit"): policy})

    with pytest.raises(MaterialisationError) as excinfo:
        select_facet(
            "unit",
            FacetBinding(state="unbound"),
            candidates=[_FACET_KEN, _FACET_TGO],
            graph=graph,
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"unit": FacetBinding(state="unbound")}, graph=graph, stat_var=_STAT_VAR
            ),
        )

    assert isinstance(excinfo.value, FacetNotAvailableError)
    assert excinfo.value.code == "facet_not_available"


def _selection(axis: str, ref: str) -> FacetSelection:
    return FacetSelection(
        axis=axis,
        ref=ref,
        basis="bound",
        default_policy=None,
        default_policy_version=None,
        selection_reason=None,
        dispatch_facet_id="unused",
    )


def test_joint_dispatch_facet_ids_finds_the_sole_joint_match_not_the_witnesses():
    # A/B/C each match exactly one axis and become that axis's witness. D
    # matches all three axes at once, so it alone lands in the joint
    # dispatch set.
    candidate_a = Facet(
        facet_id="fA",
        provenance_id="P",
        measurement_method="other",
        unit="other",
        obs_count=1,
        earliest_date=None,
        latest_date=None,
    )
    candidate_b = Facet(
        facet_id="fB",
        provenance_id="other",
        measurement_method="M",
        unit="other",
        obs_count=1,
        earliest_date=None,
        latest_date=None,
    )
    candidate_c = Facet(
        facet_id="fC",
        provenance_id="other",
        measurement_method="other",
        unit="U",
        obs_count=1,
        earliest_date=None,
        latest_date=None,
    )
    candidate_d = Facet(
        facet_id="fD",
        provenance_id="P",
        measurement_method="M",
        unit="U",
        obs_count=1,
        earliest_date=None,
        latest_date=None,
    )
    selections = {
        "provenance": _selection("provenance", "P"),
        "measurementMethod": _selection("measurementMethod", "M"),
        "unit": _selection("unit", "U"),
    }

    dispatch_ids = _dispatched([candidate_a, candidate_b, candidate_c, candidate_d], selections)

    assert dispatch_ids == ("fD",)


def test_joint_dispatch_facet_ids_empty_when_no_candidate_jointly_matches():
    candidate_a = Facet(
        facet_id="fA",
        provenance_id="P",
        measurement_method="other",
        unit="other",
        obs_count=1,
        earliest_date=None,
        latest_date=None,
    )
    selections = {
        "provenance": _selection("provenance", "P"),
        "measurementMethod": _selection("measurementMethod", "M"),
        "unit": _selection("unit", "U"),
    }

    assert _dispatched([candidate_a], selections) == ()


def test_joint_dispatch_facet_ids_excludes_a_match_with_no_facet_id():
    candidate = Facet(
        facet_id=None,
        provenance_id="P",
        measurement_method="M",
        unit="U",
        obs_count=1,
        earliest_date=None,
        latest_date=None,
    )
    selections = {
        "provenance": _selection("provenance", "P"),
        "measurementMethod": _selection("measurementMethod", "M"),
        "unit": _selection("unit", "U"),
    }

    assert _dispatched([candidate], selections) == ()


# --- `FacetBinding` state="request" construction (`__post_init__`) -------

_UNIT_REQUEST_USD_CURRENT = UnitRequest(currency="USD", prices="current")


def test_facet_binding_request_state_rejects_a_ref():
    with pytest.raises(MaterialisationUsageError):
        FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT, ref="USD")


def test_facet_binding_request_state_requires_a_request():
    with pytest.raises(MaterialisationUsageError):
        FacetBinding(state="request")


def test_select_facet_rejects_a_request_binding_for_an_axis_other_than_unit():
    # Request bindings belong on the unit axis; `select_facet` validates this
    # rule independently of `MaterialisationRequest` construction.
    binding = FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)

    with pytest.raises(MaterialisationUsageError):
        select_facet(
            "provenance",
            binding,
            candidates=[_FACET_NATIVE_USD],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"provenance": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
            ),
        )


def test_facet_binding_bound_or_unbound_state_rejects_a_request():
    with pytest.raises(MaterialisationUsageError):
        FacetBinding(state="bound", ref="USD", request=_UNIT_REQUEST_USD_CURRENT)
    with pytest.raises(MaterialisationUsageError):
        FacetBinding(state="unbound", request=_UNIT_REQUEST_USD_CURRENT)


# --- `select_facet`'s own `state="request"` branch ------------------------

_FACET_NATIVE_USD = Facet(
    facet_id="facetNativeUSD",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="CurrencyUSD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)
_FACET_NATIVE_EUR = Facet(
    facet_id="facetNativeEUR",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="CurrencyEUR",
    obs_count=1000,
    earliest_date="2020",
    latest_date="2020",
)


def test_request_prefers_native_currency_candidate_over_a_higher_ranked_alternative():
    # facetNativeEUR outranks facetNativeUSD on obs_count alone (1000 vs 1),
    # but the request's own currency preference for {USD, current} wins
    # over the ordinary ranking.
    binding = FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)

    selection = select_facet(
        "unit",
        binding,
        candidates=[_FACET_NATIVE_EUR, _FACET_NATIVE_USD],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert selection.ref == "CurrencyUSD"
    assert selection.dispatch_facet_id == "facetNativeUSD"
    assert selection.selection_reason == "native_currency_preferred"
    assert selection.request is _UNIT_REQUEST_USD_CURRENT


def test_request_selects_a_currency_unit_of_measure_facet_in_the_requested_currency():
    """Base Data Commons `USDollar` is denominated in USD, so it is the native match for a
    current-price USD request over a higher-ranked `CurrencyEUR` facet."""
    facet_us_dollar = replace(_FACET_NATIVE_USD, facet_id="facetUSDollar", unit="USDollar")
    binding = FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)

    selection = select_facet(
        "unit",
        binding,
        candidates=[_FACET_NATIVE_EUR, facet_us_dollar],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert selection.ref == "USDollar"
    assert selection.dispatch_facet_id == "facetUSDollar"
    assert selection.selection_reason == "native_currency_preferred"


def test_request_prefers_the_currency_unit_over_its_currency_unit_of_measure_twin():
    """A `CurrencyUSD` facet keeps precedence over a better-observed `USDollar` facet."""
    facet_us_dollar = replace(
        _FACET_NATIVE_USD, facet_id="facetUSDollar", unit="USDollar", obs_count=5000
    )
    binding = FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)

    selection = select_facet(
        "unit",
        binding,
        candidates=[facet_us_dollar, _FACET_NATIVE_USD],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert selection.ref == "CurrencyUSD"
    assert selection.dispatch_facet_id == "facetNativeUSD"


def test_request_falls_to_ordinary_default_when_no_native_currency_candidate():
    # With no USD candidate, ordinary ranking selects the sole viable unit,
    # `CurrencyEUR`.
    binding = FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)

    selection = select_facet(
        "unit",
        binding,
        candidates=[_FACET_NATIVE_EUR],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert selection.ref == "CurrencyEUR"
    assert selection.selection_reason == "sole_candidate"
    assert selection.request is _UNIT_REQUEST_USD_CURRENT


def test_request_role_key_currency_skips_the_preference_and_uses_ordinary_default():
    # Role-key currencies use ordinary ranking, regardless of `prices`.
    request = UnitRequest(currency="observationAbout", prices="current")
    binding = FacetBinding(state="request", request=request)

    selection = select_facet(
        "unit",
        binding,
        candidates=[_FACET_NATIVE_USD],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert selection.ref == "CurrencyUSD"
    assert selection.selection_reason == "sole_candidate"


def test_request_constant_prices_skips_the_preference_and_uses_ordinary_default():
    # Constant prices use ordinary facet ranking, even for an ISO currency.
    request = UnitRequest(currency="USD", prices="constant", base_year=2015)
    binding = FacetBinding(state="request", request=request)

    selection = select_facet(
        "unit",
        binding,
        candidates=[_FACET_NATIVE_USD],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert selection.ref == "CurrencyUSD"
    assert selection.selection_reason == "sole_candidate"


def test_request_empty_narrowed_pool_raises_facet_not_available_at_unit_not_default_undefined():
    # Bound to a different provenance than the only candidate carries, so
    # the request's own narrowed pool is empty.
    all_bindings = {"provenance": FacetBinding(state="bound", ref="OTHER_PROV")}
    binding = FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)

    with pytest.raises(MaterialisationError) as excinfo:
        select_facet(
            "unit",
            binding,
            candidates=[_FACET_NATIVE_USD],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            all_bindings=all_bindings,
            window=None,
            policies=declared_default_policies(
                all_bindings, graph=FakeGraphReader(), stat_var=_STAT_VAR
            ),
        )

    assert isinstance(excinfo.value, FacetNotAvailableError)
    assert excinfo.value.code == "facet_not_available"
    assert excinfo.value.facet_pointer == "/facets/unit"
    assert excinfo.value.alternatives is not None


def test_request_resolved_to_a_null_native_unit_raises_non_currency_unit_error():
    facet_no_unit = replace(_FACET_NATIVE_USD, unit=None)
    binding = FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)

    with pytest.raises(NonCurrencyUnitError) as excinfo:
        select_facet(
            "unit",
            binding,
            candidates=[facet_no_unit],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"unit": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
            ),
        )

    assert excinfo.value.code == "facet_binding_unsupported"
    assert excinfo.value.facet_pointer == "/facets/unit"


def test_request_resolved_to_a_non_currency_ref_raises_non_currency_unit_error():
    # The sole default candidate is bare `USD`, not a currency unit, so
    # `NonCurrencyUnitError` is raised.
    request = UnitRequest(currency="EUR", prices="current")
    binding = FacetBinding(state="request", request=request)

    with pytest.raises(NonCurrencyUnitError) as excinfo:
        select_facet(
            "unit",
            binding,
            candidates=[_FACET_KEN],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            window=None,
            policies=declared_default_policies(
                {"unit": binding}, graph=FakeGraphReader(), stat_var=_STAT_VAR
            ),
        )

    assert excinfo.value.unit == "USD"


def test_request_declared_policy_over_the_narrowed_pool_wins_over_ranking():
    policy = FacetDefaultPolicy(
        resolved_ref="CurrencyGBP",
        policy_node="policy/unit-default",
        policy_version="v1",
        selection_reason="most-widely-reported currency",
    )
    graph = FakeGraphReader(policies={(_STAT_VAR, "unit"): policy})
    facet_gbp = replace(_FACET_NATIVE_USD, facet_id="facetGBP", unit="CurrencyGBP", obs_count=1)
    # No `CurrencyUSD` candidate, so the preference check falls through and
    # the declared policy (over the narrowed pool) wins ahead of ranking.
    binding = FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)

    selection = select_facet(
        "unit",
        binding,
        candidates=[facet_gbp],
        graph=graph,
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies({"unit": binding}, graph=graph, stat_var=_STAT_VAR),
    )

    assert selection.basis == "declared_default"
    assert selection.ref == "CurrencyGBP"
    assert selection.default_policy == "policy/unit-default"
    assert selection.selection_reason == "most-widely-reported currency"
    assert selection.request is _UNIT_REQUEST_USD_CURRENT


def test_request_binding_narrows_sibling_axes_to_the_currency_preferred_pool():
    # facet_p2 leads by obs_count (1000 vs 1). An unbound unit selects it;
    # a USD/current request narrows all axes to facet_p1, keeping them coherent.
    facet_p1 = Facet(
        facet_id="f1",
        provenance_id="P1",
        measurement_method="M1",
        unit="CurrencyUSD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    facet_p2 = Facet(
        facet_id="f2",
        provenance_id="P2",
        measurement_method="M2",
        unit="CurrencyEUR",
        obs_count=1000,
        earliest_date="2015",
        latest_date="2015",
    )
    request_bound = {"unit": FacetBinding(state="request", request=_UNIT_REQUEST_USD_CURRENT)}
    unbound = {"unit": FacetBinding(state="unbound")}
    expected_ref = {"provenance": "P1", "measurementMethod": "M1"}
    expected_ref_unnarrowed = {"provenance": "P2", "measurementMethod": "M2"}

    for axis in ("provenance", "measurementMethod"):
        with_request = select_facet(
            axis,
            FacetBinding(state="unbound"),
            candidates=[facet_p1, facet_p2],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            all_bindings=request_bound,
            window=None,
            policies=declared_default_policies(
                request_bound, graph=FakeGraphReader(), stat_var=_STAT_VAR
            ),
        )
        with_unbound_unit = select_facet(
            axis,
            FacetBinding(state="unbound"),
            candidates=[facet_p1, facet_p2],
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
            all_bindings=unbound,
            window=None,
            policies=declared_default_policies(
                unbound, graph=FakeGraphReader(), stat_var=_STAT_VAR
            ),
        )
        # An unbound unit ranks the full, disagreeing pool: facet_p2 (the
        # higher-obs_count facet) wins.
        assert with_unbound_unit.ref == expected_ref_unnarrowed[axis]
        assert with_unbound_unit.selection_reason == "most_observations_over_disagreement"
        # A currency-native unit request narrows the pool to the sole
        # USD-carrying facet first, so this axis follows that narrower pool
        # to facet_p1 instead.
        assert with_request.ref == expected_ref[axis]
        assert with_request.dispatch_facet_id == "f1"
        assert with_request.selection_reason == "sole_candidate"


def test_bound_and_unbound_unit_selections_carry_no_request():
    bound_selection = select_facet(
        "unit",
        FacetBinding(state="bound", ref="USD"),
        candidates=[_FACET_KEN],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": FacetBinding(state="bound", ref="USD")},
            graph=FakeGraphReader(),
            stat_var=_STAT_VAR,
        ),
    )
    unbound_selection = select_facet(
        "unit",
        FacetBinding(state="unbound"),
        candidates=[_FACET_KEN],
        graph=FakeGraphReader(),
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(
            {"unit": FacetBinding(state="unbound")}, graph=FakeGraphReader(), stat_var=_STAT_VAR
        ),
    )

    assert bound_selection.request is None
    assert unbound_selection.request is None


def test_as_grammar_facet_ref_selection_projects_ref_and_basis():
    selection = _selection("provenance", "OECD_DAC2A")

    assert selection.as_grammar_facet() == {
        "axis": "provenance",
        "ref": "OECD_DAC2A",
        "basis": "bound",
    }


def test_as_grammar_facet_deployment_fallback_collapses_to_declared_default():
    selection = FacetSelection(
        axis="unit",
        ref="USD",
        basis="deployment_fallback",
        default_policy=DEPLOYMENT_FACET_POLICY,
        default_policy_version=DEPLOYMENT_FACET_POLICY_VERSION,
        selection_reason="sole_candidate",
        dispatch_facet_id="unused",
    )

    assert selection.as_grammar_facet() == {
        "axis": "unit",
        "ref": "USD",
        "basis": "declared_default",
    }


def test_as_grammar_facet_uniform_iso_request_projects_the_request_object():
    request = UnitRequest(currency="USD", prices="current")
    selection = FacetSelection(
        axis="unit",
        ref="CurrencyUSD",
        basis="deployment_fallback",
        default_policy=DEPLOYMENT_FACET_POLICY,
        default_policy_version=DEPLOYMENT_FACET_POLICY_VERSION,
        selection_reason="native_currency_preferred",
        dispatch_facet_id="unused",
        request=request,
    )

    assert selection.as_grammar_facet() == {
        "axis": "unit",
        "ref": {"currency": "USD", "prices": "current"},
        "basis": "bound",
    }


def test_as_grammar_facet_role_shaped_request_projects_the_role_object():
    request = UnitRequest(currency="origin", prices="current")
    selection = FacetSelection(
        axis="unit",
        ref="CurrencyEUR",
        basis="deployment_fallback",
        default_policy=DEPLOYMENT_FACET_POLICY,
        default_policy_version=DEPLOYMENT_FACET_POLICY_VERSION,
        selection_reason="sole_candidate",
        dispatch_facet_id="unused",
        request=request,
    )

    assert selection.as_grammar_facet() == {
        "axis": "unit",
        "ref": {"currency": {"role": "origin"}, "prices": "current"},
        "basis": "bound",
    }


def test_as_grammar_facet_constant_request_carries_base_year():
    request = UnitRequest(currency="USD", prices="constant", base_year=2015)
    selection = FacetSelection(
        axis="unit",
        ref="CurrencyUSD",
        basis="declared_default",
        default_policy="policy/unit-default",
        default_policy_version="v1",
        selection_reason="most-widely-reported currency",
        dispatch_facet_id="unused",
        request=request,
    )

    assert selection.as_grammar_facet() == {
        "axis": "unit",
        "ref": {"currency": "USD", "prices": "constant", "base_year": 2015},
        "basis": "bound",
    }


def _merge_pair(existing: Facet, other: Facet) -> Facet:
    """`existing` with `other`'s points added, the rule `merge_facets` applies across many."""

    def accounted(facet: Facet) -> bool:
        return facet.dates_complete and (bool(facet.dates) or facet.obs_count == 0)

    dates = existing.dates + other.dates
    latest = latest_date(existing.latest_date, other.latest_date)
    series_end = earliest_date(existing.earliest_series_end, other.earliest_series_end)
    return replace(
        existing,
        obs_count=existing.obs_count + other.obs_count,
        earliest_date=earliest_date(existing.earliest_date, other.earliest_date),
        latest_date=latest,
        first_series_end=None if series_end == latest else series_end,
        dates=dates,
        values=existing.values + other.values,
        dates_complete=(
            accounted(existing) and accounted(other)
            if dates
            else existing.dates_complete and other.dates_complete
        ),
        annual_evidence=existing.reads_annual and other.reads_annual,
    )


_MERGE_KINDS = (
    Facet(
        obs_count=2,
        earliest_date="2010",
        latest_date="2011",
        facet_id="f",
        unit="USD",
        dates=("2010", "2011"),
        values=(1.0, None),
    ),
    Facet(
        obs_count=3,
        earliest_date="2001",
        latest_date="2020",
        facet_id="f",
        unit="EUR",
        dates=("2001", "2020"),
        values=(2.0, 3.0),
        dates_complete=False,
    ),
    Facet(obs_count=4, earliest_date="1990", latest_date="2024", facet_id="f"),
    Facet(obs_count=5, earliest_date=None, latest_date="2030", facet_id="f", dates_complete=False),
    Facet(obs_count=0, earliest_date=None, latest_date=None, facet_id="f"),
)


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_merging_many_facets_equals_merging_them_one_pair_at_a_time(length: int):
    """Every ordering of dated, partly dated, undated and empty facets merges to the facet a
    pair-by-pair merge builds: the same counts, bounds, points, completeness and metadata."""
    for facets in product(_MERGE_KINDS, repeat=length):
        assert merge_facets(facets) == reduce(_merge_pair, facets), facets


def _ilo_facet(facet_id: str, period: str | None, obs_count: int) -> Facet:
    """An ILO unemployment facet for one place: the UN graph publishes the same rate as an
    annual (`P1Y`) and a quarterly (`P3M`) facet, alike on every facet axis."""
    return Facet(
        facet_id=facet_id,
        provenance_id="undata/p/ILO",
        unit="Percent",
        observation_period=period,
        obs_count=obs_count,
        earliest_date="2019",
        latest_date="2022-10" if period == "P3M" else "2022",
    )


def _selected(
    basis: Literal["bound", "deployment_fallback"] = "deployment_fallback",
) -> dict[str, FacetSelection]:
    """The selections every ILO facet carries, on `basis`."""
    return {
        axis: FacetSelection(
            axis=axis,
            ref=ref,
            basis=basis,
            default_policy=None,
            default_policy_version=None,
            selection_reason=None,
            dispatch_facet_id="unused",
        )
        for axis, ref in (
            ("provenance", "undata/p/ILO"),
            ("measurementMethod", None),
            ("unit", "Percent"),
        )
    }


def _dispatched(
    candidates: list[Facet],
    selections: dict[str, FacetSelection],
    *,
    window: RequestedWindow | None = None,
    breadth: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """The facet ids a dispatch fetches: the joint matches at the cadence the election's
    ranking decides (`facet_election`)."""
    matching = joint_matches(candidates, selections)
    election = facet_election(
        selections,
        matching=matching,
        member_matches=[matching],
        cadence_of=cadences_by_member([candidates]),
        stat_vars=["dcid:UnemploymentRate"],
        window=window,
        breadth=breadth,
    )
    return joint_dispatch_facet_ids(candidates, election, member=0)


def _dispatched_at(candidates: list[Facet], cadence: Cadence | None) -> tuple[str, ...]:
    """The facet ids a dispatch at `cadence` fetches from `candidates`."""
    return joint_dispatch_facet_ids(
        candidates,
        FacetElection(_selected(), cadence, cadences_by_member([candidates])),
        member=0,
    )


def test_joint_dispatch_facet_ids_keeps_one_observation_period():
    # #347: Kenya's ILO unemployment rate. Both facets carry the selected provenance,
    # method and unit, so dispatch fetched both, and the frame held 2019 beside 2019-01.
    quarterly = _ilo_facet("fQ", "P3M", 12)
    annual = _ilo_facet("fA", "P1Y", 3)

    assert _dispatched([quarterly, annual], _selected()) == ("fQ",)
    assert _dispatched_at([quarterly, annual], Cadence(period="P1Y")) == ("fA",)


def test_joint_dispatch_facet_ids_serves_an_unstated_annual_facet_beside_the_p1y_one():
    # One series split across a P1Y facet and one stating no period with bare-year dates
    # (Kenya's census `Count_Person`) is annual on both, so it serves whole.
    annual = _ilo_facet("fA", "P1Y", 3)
    unstated = replace(_ilo_facet("fN", None, 3), earliest_date="2020", latest_date="2022")

    assert _dispatched([annual, unstated], _selected()) == ("fA", "fN")


def test_joint_dispatch_facet_ids_reads_coverage_years_as_no_cadence_evidence():
    # A coverage candidate stating no period carries coverage's bare years
    # whatever its cadence, so it does not ride along with the elected annual facet.
    annual = _ilo_facet("fA", "P1Y", 2)
    from_coverage = replace(
        _ilo_facet("fN", None, 6), earliest_date="2019", latest_date="2020", dates_coarsened=True
    )

    assert _dispatched_at([annual, from_coverage], Cadence(period="P1Y")) == ("fA",)


@pytest.mark.parametrize("latest", [None, "2022-10"])
def test_joint_dispatch_facet_ids_takes_no_incomplete_date_list_as_annual(latest):
    # A short dated graph read keeps the dates it returned and marks them
    # incomplete (`_build_facet`), so a lone "2019" from a quarterly facet with no stated
    # period, or one whose known last date is "2022-10", is no evidence that it is annual.
    raw = {
        "facetId": "fN",
        "obsCount": 12,
        "latestDate": latest,
        "observations": [{"date": "2019", "value": 1}],
    }
    unstated = _build_facet(
        raw, {"fN": {"provenanceId": "undata/p/ILO", "unit": "Percent"}}, with_dates=True
    )
    annual = _ilo_facet("fA", "P1Y", 3)

    assert unstated.dates_complete is False
    assert _dispatched_at([annual, unstated], Cadence(period="P1Y")) == ("fA",)


def test_joint_dispatch_facet_ids_serves_a_complete_unbounded_annual_date_list_beside_p1y():
    # The read-plan case: no stated period and no bounds, but every date listed, all years.
    annual = _ilo_facet("fA", "P1Y", 3)
    unstated = replace(
        _ilo_facet("fN", None, 3),
        earliest_date=None,
        latest_date=None,
        dates=("2000", "2001", "2002"),
        dates_complete=True,
    )

    assert _dispatched([annual, unstated], _selected()) == ("fA", "fN")


@pytest.mark.parametrize(
    ("other", "elected"),
    [
        # Stated quarterly beside an unstated facet dated by month: the larger leads.
        pytest.param(_ilo_facet("fQ", "P3M", 12), "fQ", id="stated"),
        # Two unstated facets dated by month, a monthly and a quarterly one, are two
        # cadences, so only the larger dispatches.
        pytest.param(
            replace(_ilo_facet("fQ", None, 12), latest_date="2022-10"), "fQ", id="unstated"
        ),
    ],
)
def test_joint_dispatch_facet_ids_reads_an_unstated_facet_without_annual_dates_apart(
    other, elected
):
    # A facet stating no period whose dates are not bare years is a cadence of its own,
    # keyed by its facet id: it does not ride along with another facet, nor bring one along.
    unstated = replace(_ilo_facet("fN", None, 4), earliest_date="2019-01", latest_date="2019-04")

    assert _dispatched([other, unstated], _selected()) == (elected,)
    assert _dispatched_at([other, unstated], Cadence(facet_id="fN")) == ("fN",)


@pytest.mark.parametrize("order", [("fA", "fQ"), ("fQ", "fA")])
@pytest.mark.parametrize("basis", ["bound", "deployment_fallback"])
def test_joint_dispatch_facet_ids_lets_the_ranking_not_the_candidate_order_pick_the_period(
    order, basis
):
    # A bound axis's witness is only the first candidate carrying the bound value, so the
    # ranking among the joint matches picks the period, whatever the candidate order.
    by_id = {"fQ": _ilo_facet("fQ", "P3M", 12), "fA": _ilo_facet("fA", "P1Y", 50)}

    assert _dispatched([by_id[i] for i in order], _selected(basis)) == ("fA",)


@pytest.mark.parametrize(
    ("window", "breadth", "expected"),
    [
        # The window's years lead: an annual series ending in 2009 loses to the quarters.
        pytest.param(
            RequestedWindow(kind="range", start_year=2019, end_year=2026), None, "fQ", id="window"
        ),
        # Breadth leads: an annual facet covering three bound places (fewer points) beats a
        # quarterly one covering one, as the election ranks them.
        pytest.param(RequestedWindow(kind="all"), {"fA": 3, "fQ": 1}, "fA", id="breadth"),
    ],
)
def test_joint_dispatch_facet_ids_ranks_as_the_election_when_every_axis_is_bound(
    window, breadth, expected
):
    annual = replace(_ilo_facet("fA", "P1Y", 9), earliest_date="1960", latest_date="2009")
    quarterly = _ilo_facet("fQ", "P3M", 12)

    assert _dispatched([annual, quarterly], _selected("bound"), window=window, breadth=breadth) == (
        expected,
    )


def test_joint_dispatch_facet_ids_serves_one_period_when_no_witness_matches():
    # A group member whose candidates lack the group's witness.
    monthly = _ilo_facet("fM", "P1M", 40)
    quarterly = _ilo_facet("fQ", "P3M", 12)

    assert _dispatched([quarterly, monthly], _selected()) == ("fM",)


def test_joint_dispatch_facet_ids_dispatches_nothing_without_a_decided_cadence():
    assert _dispatched_at([_ilo_facet("fA", "P1Y", 3)], None) == ()


@pytest.mark.parametrize(
    ("second", "annual"),
    [
        pytest.param(("2019", "2021"), True, id="both-blocks-annual"),
        pytest.param(("2020-01", "2020-12"), False, id="one-block-monthly"),
    ],
)
def test_a_read_merging_blocks_of_one_facet_keeps_annual_evidence_only_from_every_block(
    second, annual
):
    # Two `orderedFacets` blocks of one facet id merge to the bare-year bounds `2018` to
    # `2023` either way; the merged facet reads as annual only when each block's own dates do.
    blocks = [
        {"facetId": "fN", "obsCount": 6, "earliestDate": "2018", "latestDate": "2023"},
        {"facetId": "fN", "obsCount": 3, "earliestDate": second[0], "latestDate": second[1]},
    ]

    (merged,) = _parse_ordered_facets(blocks, {"fN": {}}, with_dates=False)

    assert (merged.earliest_date, merged.latest_date) == ("2018", "2023")
    assert merged.reads_annual is annual


@pytest.mark.parametrize(
    ("dates", "annual"),
    [
        pytest.param(["2015", "2017", "2020"], True, id="every-date-a-year"),
        pytest.param(["2015", "2016-06", "2020"], False, id="a-month-inside-year-bounds"),
    ],
)
def test_a_complete_date_list_reads_as_annual_only_when_every_date_is_a_year(dates, annual):
    # The bounds are the bare years `2015` and `2020` either way; a complete dated read is
    # judged on every date it lists.
    raw = {
        "facetId": "fN",
        "obsCount": len(dates),
        "earliestDate": "2015",
        "latestDate": "2020",
        "observations": [{"date": date, "value": 1} for date in dates],
    }

    facet = _build_facet(raw, {"fN": {}}, with_dates=True)

    assert facet.dates_complete is True
    assert facet.reads_annual is annual


@pytest.mark.parametrize("facets_per_member", [1, 10])
def test_wide_group_election_bounds_cadence_lookups_by_actual_matches(facets_per_member):
    # Distinct ids with matching axes occur in index-sourced groups. The election runs
    # before fetch chunking, so it must not search every member map for every facet id.
    lookups = 0

    class CountingCadences(dict):
        def get(self, key, default=None):
            nonlocal lookups
            lookups += 1
            return super().get(key, default)

    member_count = 110
    members = [
        [
            _ilo_facet(f"member-{member}-facet-{facet}", "P1Y", 3)
            for facet in range(facets_per_member)
        ]
        for member in range(member_count)
    ]
    matching = [facet for member in members for facet in member]
    maps = tuple(CountingCadences(one) for one in cadences_by_member(members))
    election = facet_election(
        _selected(),
        matching=matching,
        member_matches=members,
        cadence_of=maps,
        stat_vars=["dcid:UnemploymentRate"],
        window=None,
        breadth=None,
    )

    assert election.cadence == Cadence(period="P1Y")
    assert election.members == frozenset(range(member_count))
    assert {selection.dispatch_facet_id for selection in election.selections.values()} == {
        "member-0-facet-0"
    }
    assert lookups <= 2 * len(matching)
