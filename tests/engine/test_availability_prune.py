"""Tests denominator-resolution pruning from helper rules through `resolve_variable`.

Only matching shapes with resolved versus unresolved denominators pair. A no-data sibling is dropped
only when its counterpart has positive evidence. Probes cover named places, regions, and entity-less
queries; unknown availability keeps both.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools

import pytest

from entity_resolution import Entity, Group
from search.engine.bind import _UniversalBindOutput
from search.engine.quantity import NO_REQUIREMENT, DenominatorKind, quantity_requirement
from search.engine.region import resolve_variable
from search.engine.region.availability import (
    _ENTITYLESS_AVAILABILITY_PROBE_PLACES,
    _PLACE_AVAILABILITY_PROBE_SAMPLE,
    _are_availability_siblings,
    _drop_shapes,
    _prune_availability_siblings,
    _resolve_sibling_availability,
    _shape_group_key,
    _sibling_probe_places,
)
from search.engine.region.codes import AVAILABILITY_PROBE_FAILED, ELECTED_SHAPE_NO_DATA
from search.engine.retrieval.derive import derive_shapes as _real_derive_shapes
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._builders import place_sets
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

# ---------------------------------------------------------------------------
# Shared shape factory
# ---------------------------------------------------------------------------


def _shape(
    shape_id: str,
    *,
    pop_type: str = "EconomicActivity",
    meas_prop: str = "amount",
    stat_type: str = "measuredValue",
    meas_qual: str | None = "Nominal",
    meas_denom_dcid: str | None = None,
    meas_denom_unresolved: bool = False,
    slot_keys: tuple[SlotKeyDraft, ...] = (),
    place_available: bool | None = None,
    declared_roles: tuple[str, ...] | None = None,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid=pop_type,
        meas_prop_dcid=meas_prop,
        stat_type_dcid=stat_type,
        meas_qual_dcid=meas_qual,
        meas_denom_dcid=meas_denom_dcid,
        slot_keys=slot_keys,
        meas_denom_unresolved=meas_denom_unresolved,
        place_available=place_available,
        declared_roles=declared_roles,
    )


def _shape_with_svs(shape: ShapeDraft, sv_dcids: list[str]) -> ShapeDraft:
    """Stamp sv_arc_facts with the given member SV dcids (empty arcs; only the
    key set matters to _resolve_sibling_availability's probe)."""
    return dataclasses.replace(shape, sv_arc_facts={key: {} for key in sv_dcids})


# ---------------------------------------------------------------------------
# _are_availability_siblings
# ---------------------------------------------------------------------------

# (meas_denom_dcid, meas_denom_unresolved) per denominator state.
_DENOM_STATES: dict[str, tuple[str | None, bool]] = {
    "absent": (None, False),
    "resolved": ("Count_Person", False),
    "unresolved": (None, True),
    # A dangling ProvisionalNode target that also carries a dcid (e.g. the
    # live "PerCapita" node). Unresolved, same as "unresolved" above, but
    # meas_denom_dcid is populated too. See FiveTuple.meas_denom_unresolved.
    "unresolved_with_dcid": ("PerCapita", True),
}

# The only two state pairings _are_availability_siblings accepts are one side
# unresolved (dcid-less or dcid-bearing) and the other resolved. Every other
# pairing, including the two unresolved variants against each other and
# "absent" paired with anything, fails the sibling test.
_SIBLING_STATE_PAIRS = {
    frozenset({"resolved", "unresolved"}),
    frozenset({"resolved", "unresolved_with_dcid"}),
}


class TestAreAvailabilitySiblings:
    @pytest.mark.parametrize("state_a,state_b", list(itertools.product(_DENOM_STATES, repeat=2)))
    def test_denominator_state_matrix(self, state_a: str, state_b: str):
        """Only resolved/unresolved denominator pairs are siblings."""
        dcid_a, unresolved_a = _DENOM_STATES[state_a]
        dcid_b, unresolved_b = _DENOM_STATES[state_b]
        a = _shape("a", meas_denom_dcid=dcid_a, meas_denom_unresolved=unresolved_a)
        b = _shape("b", meas_denom_dcid=dcid_b, meas_denom_unresolved=unresolved_b)

        expected = frozenset({state_a, state_b}) in _SIBLING_STATE_PAIRS
        assert _are_availability_siblings(a, b) is expected

    def test_identical_everything_but_denominator_resolution_are_siblings(self):
        slots = (SlotKeyDraft(axis="how", property_dcid="activitySource", label="source"),)
        a = _shape("a", slot_keys=slots, meas_denom_dcid=None, meas_denom_unresolved=True)
        b = _shape(
            "b", slot_keys=slots, meas_denom_dcid="Count_Person", meas_denom_unresolved=False
        )
        assert _are_availability_siblings(a, b) is True

    def test_qualifier_mismatch_breaks_an_otherwise_matching_pair(self):
        a = _shape("a", meas_qual="Nominal", meas_denom_dcid=None, meas_denom_unresolved=True)
        b = _shape(
            "b", meas_qual="Real", meas_denom_dcid="Count_Person", meas_denom_unresolved=False
        )
        assert _are_availability_siblings(a, b) is False

    def test_constraint_slot_key_set_mismatch_breaks_an_otherwise_matching_pair(self):
        slot_a = (SlotKeyDraft(axis="how", property_dcid="activitySource", label="source"),)
        slot_b = (SlotKeyDraft(axis="how", property_dcid="otherProp", label="other"),)
        a = _shape("a", slot_keys=slot_a, meas_denom_dcid=None, meas_denom_unresolved=True)
        b = _shape(
            "b", slot_keys=slot_b, meas_denom_dcid="Count_Person", meas_denom_unresolved=False
        )
        assert _are_availability_siblings(a, b) is False

    def test_population_type_mismatch_breaks_an_otherwise_matching_pair(self):
        a = _shape("a", pop_type="EconomicActivity", meas_denom_unresolved=True)
        b = _shape(
            "b", pop_type="Emissions", meas_denom_dcid="Count_Person", meas_denom_unresolved=False
        )
        assert _are_availability_siblings(a, b) is False

    def test_partial_five_tuple_matching_empty_fields_are_siblings(self):
        """Matching empty five-tuple fields remain equal for custom-DC imports."""
        a = _shape("a", pop_type="", meas_prop="", stat_type="", meas_denom_unresolved=True)
        b = _shape(
            "b",
            pop_type="",
            meas_prop="",
            stat_type="",
            meas_denom_dcid="Count_Person",
            meas_denom_unresolved=False,
        )
        assert _are_availability_siblings(a, b) is True

    def test_partial_five_tuple_never_pairs_with_a_populated_shape(self):
        """Empty five-tuple fields do not match populated fields as wildcards."""
        a = _shape("a", pop_type="", meas_prop="", stat_type="", meas_denom_unresolved=True)
        b = _shape(
            "b",
            pop_type="EconomicActivity",
            meas_prop="amount",
            stat_type="measuredValue",
            meas_denom_dcid="Count_Person",
            meas_denom_unresolved=False,
        )
        assert _are_availability_siblings(a, b) is False

    def test_axis_differing_same_property_still_matches(self):
        """Sibling matching compares slot properties, regardless of their classified axes."""
        slot_a = (SlotKeyDraft(axis="how", property_dcid="activitySource", label="source"),)
        slot_b = (SlotKeyDraft(axis="what", property_dcid="activitySource", label="source"),)
        a = _shape("a", slot_keys=slot_a, meas_denom_dcid=None, meas_denom_unresolved=True)
        b = _shape(
            "b", slot_keys=slot_b, meas_denom_dcid="Count_Person", meas_denom_unresolved=False
        )
        assert _are_availability_siblings(a, b) is True


# ---------------------------------------------------------------------------
# _prune_availability_siblings
# ---------------------------------------------------------------------------

_ADULTS = "Count_Person_15OrMoreYears"


class TestPruneAvailabilitySiblings:
    def test_drops_unresolved_when_its_resolved_sibling_has_data(self):
        unresolved = _shape("unresolved", meas_denom_unresolved=True, place_available=False)
        resolved = _shape("resolved", meas_denom_dcid="Count_Person", place_available=True)

        result, _ = _prune_availability_siblings(
            [unresolved, resolved], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
        )

        assert [s.shape_id for s in result] == ["resolved"]

    def test_drops_resolved_when_its_unresolved_sibling_has_data(self):
        """The unresolved sibling survives when it has data and the resolved sibling does not."""
        unresolved = _shape("unresolved", meas_denom_unresolved=True, place_available=True)
        resolved = _shape("resolved", meas_denom_dcid="Count_Person", place_available=False)

        result, _ = _prune_availability_siblings(
            [unresolved, resolved], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
        )

        assert [s.shape_id for s in result] == ["unresolved"]

    def test_keeps_both_when_the_resolved_sibling_has_no_data_either(self):
        unresolved = _shape("unresolved", meas_denom_unresolved=True, place_available=False)
        resolved = _shape("resolved", meas_denom_dcid="Count_Person", place_available=False)

        result, _ = _prune_availability_siblings(
            [unresolved, resolved], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
        )

        assert {s.shape_id for s in result} == {"unresolved", "resolved"}

    @pytest.mark.parametrize(
        ("unresolved_available", "resolved_available"),
        [
            pytest.param(None, True, id="unknown-unresolved"),
            pytest.param(True, None, id="unknown-resolved"),
        ],
    )
    def test_keeps_both_when_availability_is_unknown(
        self, unresolved_available: bool | None, resolved_available: bool | None
    ):
        unresolved = _shape(
            "unresolved",
            meas_denom_unresolved=True,
            place_available=unresolved_available,
        )
        resolved = _shape(
            "resolved",
            meas_denom_dcid="Count_Person",
            place_available=resolved_available,
        )

        result, _ = _prune_availability_siblings(
            [unresolved, resolved], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
        )

        assert {s.shape_id for s in result} == {"unresolved", "resolved"}

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            pytest.param(
                _shape("per_capita", meas_denom_dcid="Count_Person", place_available=True),
                _shape("per_sqkm", meas_denom_dcid="Area", place_available=False),
                id="different-resolved-denominators",
            ),
            pytest.param(
                _shape("aggregate", meas_denom_dcid=None, place_available=False),
                _shape("per_capita", meas_denom_dcid="Count_Person", place_available=True),
                id="denominator-presence-difference",
            ),
        ],
    )
    def test_non_sibling_denominator_pairs_are_not_pruned(self, first, second):
        result, _ = _prune_availability_siblings(
            [first, second], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
        )
        assert {shape.shape_id for shape in result} == {first.shape_id, second.shape_id}

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            pytest.param(
                _shape(
                    "role_bearing",
                    meas_denom_unresolved=True,
                    place_available=False,
                    declared_roles=("creditor", "debtor"),
                ),
                _shape("resolved", meas_denom_dcid="Count_Person", place_available=True),
                id="role-bearing-unresolved",
            ),
            pytest.param(
                _shape("unresolved", meas_denom_unresolved=True, place_available=False),
                _shape(
                    "role_bearing_other",
                    meas_denom_dcid="Count_Person",
                    place_available=True,
                    declared_roles=("creditor", "debtor"),
                ),
                id="role-bearing-resolved",
            ),
            pytest.param(
                _shape("resolved", meas_denom_dcid="Count_Person", place_available=False),
                _shape(
                    "role_bearing_other",
                    meas_denom_unresolved=True,
                    place_available=True,
                    declared_roles=("creditor", "debtor"),
                ),
                id="role-bearing-unresolved-mirrored",
            ),
        ],
    )
    def test_role_bearing_pairs_are_not_pruned(self, first, second):
        result, _ = _prune_availability_siblings(
            [first, second], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
        )
        assert {shape.shape_id for shape in result} == {first.shape_id, second.shape_id}

    @pytest.mark.parametrize(
        ("kinds", "survivors"),
        [
            pytest.param({_ADULTS: "person"}, {"per_adult"}, id="person-count"),
            pytest.param({_ADULTS: "other"}, {"unresolved", "per_adult"}, id="other-series"),
        ],
    )
    def test_a_per_capita_leg_substitutes_the_sibling_its_kinds_read_as_per_person(
        self, kinds: dict[str, DenominatorKind], survivors: set[str]
    ):
        """The prune judges the sibling's denominator as bind and materialise do: a
        `Count_Person_15OrMoreYears` sibling with data replaces the empty shape of a leg
        naming "per capita" only when the leg's kinds read it as a count of persons."""
        variable = "alcohol consumption per capita"
        empty = _shape("unresolved", meas_denom_unresolved=True, place_available=False)
        per_adult = _shape("per_adult", meas_denom_dcid=_ADULTS, place_available=True)

        result, _ = _prune_availability_siblings(
            [empty, per_adult],
            named_place=True,
            variable=variable,
            quantity=quantity_requirement([variable]),
            denominator_kinds=kinds,
        )

        assert {s.shape_id for s in result} == survivors


# ---------------------------------------------------------------------------
# _drop_shapes: never-prune-to-empty guard
# ---------------------------------------------------------------------------


class TestDropShapesNeverEmpty:
    def test_dropping_every_shape_returns_the_original_list_unchanged(self):
        shapes = [_shape("a"), _shape("b")]

        result = _drop_shapes(shapes, {_shape_group_key(shape) for shape in shapes})

        assert result == shapes

    def test_partial_drop_removes_only_the_named_shapes(self):
        shapes = [_shape("a"), _shape("b"), _shape("c")]

        result = _drop_shapes(shapes, {_shape_group_key(shapes[1])})

        assert [s.shape_id for s in result] == ["a", "c"]

    def test_empty_to_drop_is_a_no_op(self):
        shapes = [_shape("a")]

        assert _drop_shapes(shapes, set()) is shapes


# ---------------------------------------------------------------------------
# _sibling_probe_places
# ---------------------------------------------------------------------------


class TestSiblingProbePlaces:
    def test_named_place_alone_is_the_scope(self):
        assert _sibling_probe_places(["country/KEN"], {}) == ["country/KEN"]

    def test_region_children_are_sampled_and_unioned_with_named_places(self):
        result = _sibling_probe_places(
            ["country/KEN"],
            place_sets(regions={"Africa": ("africa", ["country/A1", "country/A2"])}),
        )
        assert result == ["country/KEN", "africa", "country/A1", "country/A2"]

    def test_a_group_with_no_node_gives_its_member_sample_alone(self):
        result = _sibling_probe_places(
            None, place_sets(groups={"Union": ["country/A1", "country/A2"]})
        )
        assert result == ["country/A1", "country/A2"]

    def test_region_sample_is_bounded_per_region(self):
        many_children = [f"country/C{i}" for i in range(_PLACE_AVAILABILITY_PROBE_SAMPLE + 5)]

        result = _sibling_probe_places(
            None, place_sets(regions={"Africa": ("africa", many_children)})
        )

        assert result == ["africa", *many_children[:_PLACE_AVAILABILITY_PROBE_SAMPLE]]

    def test_multiple_regions_each_contribute_their_own_bounded_sample(self):
        result = _sibling_probe_places(
            None,
            place_sets(
                regions={
                    "Africa": ("africa", ["country/A1"]),
                    "Asia": ("asia", ["country/S1", "country/S2"]),
                }
            ),
        )
        assert result == ["africa", "country/A1", "asia", "country/S1", "country/S2"]

    def test_no_named_place_and_no_region_falls_back_to_the_fixed_sample(self):
        assert _sibling_probe_places(None, {}) == list(_ENTITYLESS_AVAILABILITY_PROBE_PLACES)
        assert _sibling_probe_places([], {}) == list(_ENTITYLESS_AVAILABILITY_PROBE_PLACES)

    def test_duplicate_places_across_named_and_region_scope_are_deduped(self):
        result = _sibling_probe_places(
            ["country/A1"],
            place_sets(regions={"Africa": ("africa", ["country/A1", "country/A2"])}),
        )
        assert result == ["country/A1", "africa", "country/A2"]

    def test_childless_region_contributes_its_own_dcid(self):
        """A childless region probes its own dcid."""
        result = _sibling_probe_places(None, place_sets(regions={"Region": ("region", [])}))
        assert result == ["region"]
        assert result != list(_ENTITYLESS_AVAILABILITY_PROBE_PLACES)

    def test_unresolvable_name_refuses_loudly(self):
        """A memberless set without a node raises instead of producing a bad probe."""
        with pytest.raises(AssertionError):
            _sibling_probe_places(None, place_sets(groups={"Region": []}))


# ---------------------------------------------------------------------------
# _resolve_sibling_availability
# ---------------------------------------------------------------------------


class _CountingObsGraph(FakeGraph):
    """Records StatVars sent to `observation_facets_batch`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[list[str]] = []

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False):
        self.calls.append(list(stat_vars))
        return super().observation_facets_batch(stat_vars, entities, needs_dates=needs_dates)


class TestResolveSiblingAvailability:
    @pytest.mark.parametrize(
        "declared_roles",
        [
            pytest.param(None, id="default-declaration"),
            pytest.param(("observationAbout",), id="explicit-subject-only-declaration"),
        ],
    )
    def test_stamps_true_on_the_side_with_data_and_false_on_the_side_without(
        self, declared_roles: tuple[str, ...] | None
    ):
        unresolved = _shape(
            "unresolved",
            meas_denom_unresolved=True,
            declared_roles=declared_roles,
        )
        resolved = _shape(
            "resolved",
            meas_denom_dcid="Count_Person",
            declared_roles=declared_roles,
        )
        graph = FakeGraph(
            obs={f"{unresolved.shape_id}|country/KEN": [{"obsCount": 5}]},
            nodes={},
        )
        # sv_arc_facts is keyed by SV dcid; reuse each shape's own shape_id as
        # its representative SV dcid so the obs fixture key above matches.
        unresolved = _shape_with_svs(unresolved, [unresolved.shape_id])
        resolved = _shape_with_svs(resolved, [resolved.shape_id])

        result, probe_failed = _resolve_sibling_availability(
            [unresolved, resolved], graph=graph, named_place=True, probe_places=["country/KEN"]
        )

        assert probe_failed is False
        by_id = {s.shape_id: s for s in result}
        assert by_id["unresolved"].place_available is True
        assert by_id["resolved"].place_available is False

    def test_a_sets_probe_sees_data_held_only_at_its_node_or_only_at_its_members(self):
        """One sibling holds data at the region's node alone and the other at a member
        alone. The probe over the region's places finds both, each at its own place."""
        at_node = _shape_with_svs(_shape("at_node", meas_denom_unresolved=True), ["sv/node"])
        at_members = _shape_with_svs(
            _shape("at_members", meas_denom_dcid="Count_Person"), ["sv/members"]
        )
        graph = FakeGraph(
            obs={
                "sv/node|africa": [{"obsCount": 5}],
                "sv/members|country/A1": [{"obsCount": 5}],
            },
            nodes={},
        )

        result, probe_failed = _resolve_sibling_availability(
            [at_node, at_members],
            graph=graph,
            named_place=True,
            probe_places=_sibling_probe_places(
                None, place_sets(regions={"Africa": ("africa", ["country/A1", "country/A2"])})
            ),
        )

        assert probe_failed is False
        assert [shape.available_places for shape in result] == [{"africa"}, {"country/A1"}]

    def test_distinct_unresolved_denominators_keep_their_own_probe_results(self):
        unresolved_with_dcid = _shape_with_svs(
            _shape(
                "per_unresolved",
                meas_denom_dcid="PerCapita",
                meas_denom_unresolved=True,
            ),
            ["sv/unresolved-with-dcid"],
        )
        unresolved_without_dcid = _shape_with_svs(
            _shape("per_unresolved", meas_denom_unresolved=True),
            ["sv/unresolved-without-dcid"],
        )
        resolved = _shape_with_svs(
            _shape("per_count_person", meas_denom_dcid="Count_Person"),
            ["sv/resolved"],
        )
        graph = FakeGraph(
            obs={"sv/unresolved-with-dcid|country/KEN": [{"obsCount": 5}]},
            nodes={},
        )

        result, probe_failed = _resolve_sibling_availability(
            [unresolved_with_dcid, unresolved_without_dcid, resolved],
            graph=graph,
            named_place=True,
            probe_places=["country/KEN"],
        )

        assert probe_failed is False
        assert [tuple((shape.sv_arc_facts or {}).keys()) for shape in result] == [
            ("sv/unresolved-with-dcid",),
            ("sv/unresolved-without-dcid",),
            ("sv/resolved",),
        ]
        assert [shape.place_available for shape in result] == [True, False, False]

    def test_a_shape_with_no_sibling_in_the_set_is_left_untouched(self):
        lone = _shape("lone", meas_denom_unresolved=True)
        lone = _shape_with_svs(lone, [lone.shape_id])
        graph = FakeGraph(obs={f"{lone.shape_id}|country/KEN": [{"obsCount": 5}]}, nodes={})

        result, probe_failed = _resolve_sibling_availability(
            [lone], graph=graph, named_place=True, probe_places=["country/KEN"]
        )

        assert probe_failed is False
        assert result[0].place_available is None

    def test_graph_error_resets_place_available_to_unknown_and_reports_failure(self):
        """A failed supplemental probe resets both sibling verdicts to unknown and reports failure."""
        unresolved = _shape("unresolved", meas_denom_unresolved=True, place_available=True)
        resolved = _shape("resolved", meas_denom_dcid="Count_Person", place_available=False)
        unresolved = _shape_with_svs(unresolved, [unresolved.shape_id])
        resolved = _shape_with_svs(resolved, [resolved.shape_id])
        graph = FakeGraph(raise_on_call=True)

        result, probe_failed = _resolve_sibling_availability(
            [unresolved, resolved], graph=graph, named_place=True, probe_places=["country/KEN"]
        )

        assert probe_failed is True
        assert all(s.place_available is None for s in result)

    def test_probe_covers_only_candidate_sibling_pairs_not_every_shape(self):
        unresolved = _shape_with_svs(_shape("unresolved", meas_denom_unresolved=True), ["sv_u"])
        resolved = _shape_with_svs(_shape("resolved", meas_denom_dcid="Count_Person"), ["sv_r"])
        lone = _shape_with_svs(_shape("lone", meas_qual="Real"), ["sv_lone"])
        graph = _CountingObsGraph(obs={})

        _resolve_sibling_availability(
            [unresolved, resolved, lone],
            graph=graph,
            named_place=True,
            probe_places=["country/KEN"],
        )

        assert graph.calls, "observation_facets_batch was never called"
        assert set(graph.calls[0]) == {"sv_u", "sv_r"}

    def test_no_probe_places_is_a_no_op(self):
        unresolved = _shape("unresolved", meas_denom_unresolved=True)
        resolved = _shape("resolved", meas_denom_dcid="Count_Person")
        graph = _CountingObsGraph(obs={})

        result, probe_failed = _resolve_sibling_availability(
            [unresolved, resolved], graph=graph, named_place=True, probe_places=[]
        )

        assert not graph.calls
        assert probe_failed is False
        assert all(s.place_available is None for s in result)


# ---------------------------------------------------------------------------
# resolve_variable end-to-end: two siblings, one empty
# ---------------------------------------------------------------------------

# A GDP-per-capita-shaped pair: one SV whose measurementDenominator resolves
# to a dcid, one whose denominator arc is present but unresolved (a
# dcid-less ProvisionalNode, mirroring a real dfo-dev arc -- see
# test_discover_facts.py's TestMeasDenomTriState).
_RESOLVED_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita"
_UNRESOLVED_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita_Unresolved"
_DETECT_QUERY = "GDP per capita"


def _gdp_arcs(denominator: dict | None) -> dict:
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if denominator is not None:
        arcs["measurementDenominator"] = denominator
    return arcs


_NODES: dict = {
    _RESOLVED_SV: {
        "label": "GDP as a fraction of population",
        "type": "StatisticalVariable",
        "arcs": _gdp_arcs({"nodes": [{"dcid": "Count_Person"}]}),
    },
    _UNRESOLVED_SV: {
        "label": "GDP per capita",
        "type": "StatisticalVariable",
        "arcs": _gdp_arcs(
            {"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]}
        ),
    },
    "country/VNM": {"label": "Vietnam", "type": "Country"},
    "boreal": {"label": "Boreal", "type": "Continent", "arcs": {}},
}
_DETECT: dict = {_DETECT_QUERY: {"svs": [_RESOLVED_SV, _UNRESOLVED_SV]}}
_RESOLVE: dict = {"Vietnam": "country/VNM"}
_PLACES: dict = {"Boreal": {"dcid": "boreal", "type_of": "Continent"}}
_CHILDREN: dict = {
    "boreal": {"Country": ["country/B1"]},
    "Earth": {"Country": ["country/B1", "country/KEN", "country/USA", "country/VNM"]},
}
_OBS: dict = {
    f"{_UNRESOLVED_SV}|country/VNM": [
        {"earliestDate": "2023", "latestDate": "2023", "obsCount": 40}
    ],
    # _RESOLVED_SV carries no observations at country/VNM at all -- the
    # availability probe runs and confirms it, rather than never resolving
    # a verdict.
}


class _UnusedLLM:
    """Placeholder llm= argument for resolve_variable calls whose bind step
    is replaced by the _capture_bind_shapes spy -- it must never actually
    be invoked."""

    async def generate_structured(self, *, prompt, system, schema):
        raise AssertionError("bind_universal was replaced by a spy; the LLM must not be called")


def _capture_bind_shapes(monkeypatch) -> list[list[ShapeDraft]]:
    """Replace bind_universal with a spy that records the candidate shapes
    it was called with and elects the first one unconditionally, so the
    prune's effect on the candidate set reaching bind is directly
    observable without needing an LLM fixture for the (possibly novel)
    prompt those candidates produce."""
    captured: list[list[ShapeDraft]] = []

    async def _spy(
        variable,
        shapes,
        *,
        admitted_qualifiers=(),
        admitted_ties=(),
        withheld_measure_axes=frozenset(),
        dropped_qualifier_phrases=(),
        llm,
    ):
        captured.append(list(shapes))
        return (
            _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=None, ask=None),
            None,
        )

    monkeypatch.setattr("search.engine.region.phase_rank_bind.bind_universal", _spy)
    return captured


def _run(
    entities: list[str],
    *,
    obs: dict | None = None,
    graph_cls: type[FakeGraph] = FakeGraph,
    pre_resolved: dict | None = None,
):
    graph = graph_cls(
        nodes=_NODES,
        detect=_DETECT,
        resolve=_RESOLVE,
        obs=obs if obs is not None else _OBS,
        places=_PLACES,
        children=_CHILDREN,
    )
    return asyncio.run(
        resolve_variable(
            _DETECT_QUERY,
            entities=entities,
            date_request=None,
            detect_query=_DETECT_QUERY,
            role_query=f"{_DETECT_QUERY} of {' and '.join(entities)}",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_UnusedLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            pre_resolved=pre_resolved,
            coverage=empty_coverage_snapshot(),
        )
    )


def test_two_siblings_one_empty_offers_a_single_shape(monkeypatch):
    """The Vietnam-shaped scenario: an unresolved-denominator sibling with
    data beside a resolved-denominator sibling with none. Only the
    unresolved shape reaches bind_universal."""
    captured = _capture_bind_shapes(monkeypatch)

    _run(["Vietnam"])

    assert captured, "bind_universal was never called"
    assert len(captured[0]) == 1
    assert captured[0][0].meas_denom_unresolved is True


def test_region_expansion_prunes_using_the_combined_probe_scope(monkeypatch):
    """Naming a region (Boreal, which expands to member countries) alongside
    a resolved place probes the named place and a sample of the region's own
    members before pruning, rather than judging on the named place alone --
    mirrors a real 'GDP per capita in Kenya and Africa' query. The same
    positive evidence that prunes the Vietnam-only case above still prunes
    here, now over the full combined scope."""
    captured = _capture_bind_shapes(monkeypatch)

    _run(["Vietnam", "Boreal"])

    assert captured, "bind_universal was never called"
    assert len(captured[0]) == 1
    assert captured[0][0].meas_denom_unresolved is True


def _two_measure_head_drops(
    monkeypatch, *, elected: bool = True
) -> list[tuple[ShapeDraft, ShapeDraft]]:
    """Make the prune report two empty measure-head shapes dropped for one survivor, as
    when two rate spellings of one measure have no data at the named place. With
    `elected` false the survivor is a shape other than the one bind elects."""
    substitutions: list[tuple[ShapeDraft, ShapeDraft]] = []

    def _prune(
        shapes,
        *,
        named_place,
        quantity,
        denominator_kinds,
        variable="",
        qualifier_matches=(),
        qualifier_readings=(),
    ):
        survivors, _ = _prune_availability_siblings(
            shapes,
            named_place=named_place,
            variable=variable,
            quantity=quantity,
            denominator_kinds=denominator_kinds,
            qualifier_matches=qualifier_matches,
            qualifier_readings=qualifier_readings,
        )
        survivor = survivors[0]
        if not elected:
            survivor = dataclasses.replace(survivor, shape_id="growth", label="Growth rate")
        substitutions[:] = [
            (
                dataclasses.replace(survivor, shape_id=f"empty_{i}", label=f"Empty rate {i}"),
                survivor,
            )
            for i in (1, 2)
        ]
        return survivors, list(substitutions)

    monkeypatch.setattr("search.engine.region.phase_rank_bind._prune_availability_siblings", _prune)
    return substitutions


def test_two_measure_head_drops_warn_elected_shape_no_data_once(monkeypatch):
    """The warning describes the one election, not each shape it dropped."""
    _capture_bind_shapes(monkeypatch)
    _two_measure_head_drops(monkeypatch)

    result = _run(["Vietnam"])

    assert [w.code for w in result.warnings].count(ELECTED_SHAPE_NO_DATA) == 1


def test_measure_head_drop_warning_names_the_empty_and_elected_readings(monkeypatch):
    _capture_bind_shapes(monkeypatch)
    substitutions = _two_measure_head_drops(monkeypatch)

    result = _run(["Vietnam"])

    (warning,) = (w for w in result.warnings if w.code == ELECTED_SHAPE_NO_DATA)
    survivor = substitutions[0][1]
    assert warning.params == {
        "no_data": ["Empty rate 1", "Empty rate 2"],
        "elected": [survivor.label],
    }


def test_measure_head_drop_beside_another_election_does_not_warn(monkeypatch):
    """The prune drops empty shapes before bind. When bind elects a shape other than the
    sibling that stayed, no empty shape was the chosen reading, so nothing is warned: live
    "GDP of Kenya" drops two empty GDP growth rates and answers nominal GDP."""
    _capture_bind_shapes(monkeypatch)
    _two_measure_head_drops(monkeypatch, elected=False)

    result = _run(["Vietnam"])

    assert ELECTED_SHAPE_NO_DATA not in [w.code for w in result.warnings]


class _FailOnSecondObsCall(FakeGraph):
    """FakeGraph wrapper whose observation_facets_batch succeeds on the first
    call (derive_shapes' own availability floor) and raises on the second
    (the sibling-availability probe), then behaves normally again -- so a
    test can exercise a floor-stamped verdict beside a failed supplemental
    probe without also breaking the later materialise() calls the elected
    shape still needs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False):
        self.call_count += 1
        if self.call_count == 2:
            raise RuntimeError("simulated sibling-availability probe failure")
        return super().observation_facets_batch(stat_vars, entities, needs_dates=needs_dates)


def test_region_expansion_survives_a_failed_supplemental_probe(monkeypatch):
    """Vietnam alone already gives derive_shapes' own floor enough to stamp
    the resolved-denominator sibling place_available=False (no VNM
    observations for it) -- a verdict scoped only to Vietnam, not Boreal's
    members. When the supplemental probe over the full Vietnam+Boreal scope
    then fails, that stale False must not reach the unconditional prune:
    both siblings reset to unknown and survive to bind_universal, and the
    failure is reported as a warning."""
    captured = _capture_bind_shapes(monkeypatch)

    result = _run(["Vietnam", "Boreal"], graph_cls=_FailOnSecondObsCall)

    assert captured, "bind_universal was never called"
    assert len(captured[0]) == 2
    assert any(w.code == AVAILABILITY_PROBE_FAILED for w in result.warnings)


def test_region_expansion_keeps_both_when_the_combined_scope_has_no_data(monkeypatch):
    """The mirror of the case above: when neither sibling has any observation
    across the named place or the region's sampled members, the supplemental
    probe finds no evidence for either side and both stay unpruned."""
    captured = _capture_bind_shapes(monkeypatch)

    _run(["Vietnam", "Boreal"], obs={})

    assert captured, "bind_universal was never called"
    assert len(captured[0]) == 2


_OBS_REGION_ONLY = {
    f"{_UNRESOLVED_SV}|country/B1": [{"earliestDate": "2023", "latestDate": "2023", "obsCount": 12}]
}


def test_region_only_query_prunes_using_the_region_member_sample(monkeypatch):
    """No named country at all, just the region ("Boreal" alone) -- the probe
    scope comes entirely from the region's own expanded members
    (country/B1), since there is no named place to union it with."""
    captured = _capture_bind_shapes(monkeypatch)

    _run(["Boreal"], obs=_OBS_REGION_ONLY)

    assert captured, "bind_universal was never called"
    assert len(captured[0]) == 1
    assert captured[0][0].meas_denom_unresolved is True


_OBS_ENTITYLESS_POSITIVE = {
    f"{_UNRESOLVED_SV}|country/KEN": [
        {"earliestDate": "1960", "latestDate": "2024", "obsCount": 65}
    ]
}


def test_entityless_query_prunes_using_the_fixed_representative_sample(monkeypatch):
    """No place named at all ('GDP per capita' alone) -- the probe falls back
    to the fixed representative sample (country/USA, country/KEN), and a
    positive observation there is enough to prune the empty sibling, matching
    the gdp-per-capita-nominal golden's expected result."""
    captured = _capture_bind_shapes(monkeypatch)

    _run([], obs=_OBS_ENTITYLESS_POSITIVE)

    assert captured, "bind_universal was never called"
    assert len(captured[0]) == 1
    assert captured[0][0].meas_denom_unresolved is True


def test_entityless_query_keeps_both_when_the_representative_sample_has_no_data(monkeypatch):
    """When neither sibling has any observation at either representative
    place, the probe is inconclusive for both sides and neither is pruned:
    an inconclusive result carries no positive evidence for either side."""
    captured = _capture_bind_shapes(monkeypatch)

    _run([], obs={})

    assert captured, "bind_universal was never called"
    assert len(captured[0]) == 2


# ---------------------------------------------------------------------------
# resolve_variable end-to-end: a settled Group behaves exactly like an
# expanded region in the availability floor and the sibling probe -- it
# contributes nothing to recall_only_place_dcids (the floor's own input),
# and its members reach the sibling probe only through place_children_by_
# name's own _PLACE_AVAILABILITY_PROBE_SAMPLE-bounded sample, the same route
# a region's members already use.
# ---------------------------------------------------------------------------

_GROUP_MEMBERS = tuple(
    Entity(dcid=f"country/T{i}", name=f"Testland {i}", kind="country", graph_type="Country")
    for i in range(1, 8)
)  # 7 members: T1-T5 are _PLACE_AVAILABILITY_PROBE_SAMPLE's own sample
# (a group's place set keeps Group.members' own order),
# T6 and T7 sit past the cap.

_OBS_GROUP_MEMBER_PAST_THE_SAMPLE = {
    f"{_UNRESOLVED_SV}|country/T6": [{"earliestDate": "2023", "latestDate": "2023", "obsCount": 9}]
}
_OBS_GROUP_MEMBER_WITHIN_THE_SAMPLE = {
    f"{_UNRESOLVED_SV}|country/T3": [{"earliestDate": "2023", "latestDate": "2023", "obsCount": 9}]
}


def _capture_derive_shapes_place_dcids(monkeypatch) -> list[list[str] | None]:
    """Spy on derive_shapes' own place_dcids kwarg -- the availability
    floor's own input, region/state.py's recall_only_place_dcids
    snapshot -- without disturbing derive_shapes' real behaviour."""
    captured: list[list[str] | None] = []

    def _spy(*, place_dcids=None, **kwargs):
        captured.append(place_dcids)
        return _real_derive_shapes(place_dcids=place_dcids, **kwargs)

    monkeypatch.setattr("search.engine.region.phase_intake.derive_shapes", _spy)
    return captured


def test_settled_group_contributes_nothing_to_the_availability_floor(monkeypatch):
    """A settled Group's members must not reach derive_shapes' own floor
    probe, exactly as an expanded region's own members never do:
    recall_only_place_dcids is None, not a 7-place list."""
    captured = _capture_derive_shapes_place_dcids(monkeypatch)
    _capture_bind_shapes(monkeypatch)

    _run(
        ["TestGroup"],
        obs={},
        pre_resolved={"TestGroup": Group(members=_GROUP_MEMBERS, entity=None)},
    )

    assert captured, "derive_shapes was never called"
    assert captured[0] is None


def test_group_members_reach_the_sibling_probe_only_through_the_sampled_route(monkeypatch):
    """A group's members reach the sibling-availability probe only via
    its place set's own 5-place sample, not the full 7-member set:
    data at the 6th member (past the sample) is invisible to the probe and
    neither sibling is pruned; the same shape of data at the 3rd member
    (inside the sample) is seen and prunes the resolved-denominator sibling
    -- mirrors test_region_only_query_prunes_using_the_region_member_sample
    above, now for a Group instead of a region."""
    captured = _capture_bind_shapes(monkeypatch)
    pre_resolved = {"TestGroup": Group(members=_GROUP_MEMBERS, entity=None)}

    _run(["TestGroup"], obs=_OBS_GROUP_MEMBER_PAST_THE_SAMPLE, pre_resolved=pre_resolved)
    assert captured, "bind_universal was never called"
    assert len(captured[-1]) == 2, "past-the-sample data must not prune either sibling"

    _run(["TestGroup"], obs=_OBS_GROUP_MEMBER_WITHIN_THE_SAMPLE, pre_resolved=pre_resolved)
    assert len(captured[-1]) == 1
    assert captured[-1][0].meas_denom_unresolved is True


# ---------------------------------------------------------------------------
# Two denominator spellings that both hold data
# ---------------------------------------------------------------------------


class TestPruneSiblingCoveringFewerPlaces:
    """Nominal GDP per `Count_Person` holds one of the six sample countries and the
    `PerCapita` spelling holds all six, so a question naming no place is offered one."""

    def _pair(self, unresolved_places: set[str], resolved_places: set[str] | None):
        unresolved = dataclasses.replace(
            _shape("unresolved", meas_denom_unresolved=True, place_available=True),
            available_places=frozenset(unresolved_places),
        )
        resolved = dataclasses.replace(
            _shape("resolved", meas_denom_dcid="Count_Person", place_available=True),
            available_places=None if resolved_places is None else frozenset(resolved_places),
        )
        return [resolved, unresolved]

    def test_drops_the_spelling_whose_data_covers_a_proper_subset_of_the_places(self):
        shapes = self._pair({"country/USA", "country/DEU", "country/KEN"}, {"country/DEU"})

        result, substitutions = _prune_availability_siblings(
            shapes, named_place=False, quantity=NO_REQUIREMENT, denominator_kinds={}
        )

        assert [s.shape_id for s in result] == ["unresolved"]
        assert substitutions == []

    def test_keeps_both_when_each_holds_a_place_the_other_lacks(self):
        shapes = self._pair({"country/USA", "country/DEU"}, {"country/DEU", "country/KEN"})

        result, _ = _prune_availability_siblings(
            shapes, named_place=False, quantity=NO_REQUIREMENT, denominator_kinds={}
        )

        assert {s.shape_id for s in result} == {"unresolved", "resolved"}

    def test_keeps_both_when_they_hold_the_same_places(self):
        shapes = self._pair({"country/DEU"}, {"country/DEU"})

        result, _ = _prune_availability_siblings(
            shapes, named_place=False, quantity=NO_REQUIREMENT, denominator_kinds={}
        )

        assert {s.shape_id for s in result} == {"unresolved", "resolved"}

    def test_keeps_both_when_one_sides_places_are_unknown(self):
        shapes = self._pair({"country/USA", "country/DEU"}, None)

        result, _ = _prune_availability_siblings(
            shapes, named_place=False, quantity=NO_REQUIREMENT, denominator_kinds={}
        )

        assert {s.shape_id for s in result} == {"unresolved", "resolved"}

    def test_the_probe_records_the_places_each_shape_holds_data_at(self):
        unresolved = _shape_with_svs(_shape("unresolved", meas_denom_unresolved=True), ["wide"])
        resolved = _shape_with_svs(_shape("resolved", meas_denom_dcid="Count_Person"), ["thin"])
        graph = FakeGraph(
            obs={
                "wide|country/KEN": [{"obsCount": 5}],
                "wide|country/DEU": [{"obsCount": 5}],
                "thin|country/DEU": [{"obsCount": 5}],
            },
            nodes={},
        )

        result, _ = _resolve_sibling_availability(
            [unresolved, resolved],
            graph=graph,
            named_place=False,
            probe_places=["country/KEN", "country/DEU", "country/USA"],
        )

        by_id = {s.shape_id: s.available_places for s in result}
        assert by_id == {
            "unresolved": frozenset({"country/KEN", "country/DEU"}),
            "resolved": frozenset({"country/DEU"}),
        }
        pruned, _ = _prune_availability_siblings(
            result, named_place=False, quantity=NO_REQUIREMENT, denominator_kinds={}
        )
        assert [s.shape_id for s in pruned] == ["unresolved"]
