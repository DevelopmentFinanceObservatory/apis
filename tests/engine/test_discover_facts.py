"""Tests graph-fact readers for schema, constraints, roles, taxonomy, and identity."""

from __future__ import annotations

import pytest

from materialise.roles import DegenerateObservationProperties
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import (
    FiveTuple,
    read_constraints,
    read_declared_roles,
    read_five_tuple,
    read_slot_taxonomy,
    read_unresolved_constraints,
    standard_shape_id_for,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.shape import identity_shape_for
from tests.engine._discover_fixtures import (
    _TEST_SV_NAMESPACE,
    MEAS_PROP_DCID,
    POP_TYPE_DCID,
    PROP_PURPOSE,
    PROP_RECIPIENT,
    PROP_SCHEME,
    STAT_TYPE_DCID,
    _fake_graph_with_svs,
    _three_constraint_arcs,
)
from tests.fixtures import FakeGraph


class TestReadFiveTuple:
    def test_three_constraint_five_tuple(self):
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        ft = read_five_tuple(arcs)
        assert ft.pop_type_dcid == POP_TYPE_DCID
        assert ft.meas_prop_dcid == MEAS_PROP_DCID
        assert ft.stat_type_dcid == STAT_TYPE_DCID
        assert ft.meas_qual_dcid is None
        assert ft.meas_denom_dcid is None

    def test_missing_optional_fields_are_none(self):
        arcs = {
            "populationType": {"nodes": [{"dcid": "SomePop"}]},
            "measuredProperty": {"nodes": [{"dcid": "SomeProp"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        ft = read_five_tuple(arcs)
        assert ft.meas_qual_dcid is None
        assert ft.meas_denom_dcid is None

    def test_empty_arcs_returns_empty_strings(self):
        ft = read_five_tuple({})
        assert ft.pop_type_dcid == ""
        assert ft.meas_prop_dcid == ""


class TestMeasDenomTriState:
    """Denominators distinguish absent, unresolved without a dcid,
    unresolved with a dcid, and resolved.

    A ``ProvisionalNode`` qualifier reads as absent, even when it carries a dcid."""

    def _base_arcs(self, **extra: dict) -> dict:
        return {
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            **extra,
        }

    def test_absent_arc_is_not_unresolved(self):
        ft = read_five_tuple(self._base_arcs())
        assert ft.meas_denom_dcid is None
        assert ft.meas_denom_unresolved is False

    def test_dcid_less_provisional_node_denominator_is_unresolved(self):
        arcs = self._base_arcs(
            measurementDenominator={
                "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
            }
        )
        ft = read_five_tuple(arcs)
        assert ft.meas_denom_dcid is None
        assert ft.meas_denom_unresolved is True

    def test_dcid_bearing_provisional_node_denominator_is_unresolved(self):
        """A dcid-bearing ``ProvisionalNode`` has unresolved status and retains its dcid."""
        arcs = self._base_arcs(
            measurementDenominator={
                "nodes": [
                    {
                        "dcid": "PerCapita",
                        "types": ["ProvisionalNode"],
                        "provenanceId": "dc/base/Schema",
                    }
                ]
            }
        )
        ft = read_five_tuple(arcs)
        assert ft.meas_denom_dcid == "PerCapita"
        assert ft.meas_denom_unresolved is True

    def test_resolved_dcid_denominator_is_not_unresolved(self):
        arcs = self._base_arcs(measurementDenominator={"nodes": [{"dcid": "Count_Person"}]})
        ft = read_five_tuple(arcs)
        assert ft.meas_denom_dcid == "Count_Person"
        assert ft.meas_denom_unresolved is False

    def test_dcid_bearing_provisional_node_qualifier_reads_as_absent(self):
        """A ``ProvisionalNode`` qualifier reads as absent even when it has a dcid."""
        arcs = self._base_arcs(
            measurementQualifier={
                "nodes": [
                    {
                        "dcid": "SomeDanglingQualifier",
                        "types": ["ProvisionalNode"],
                        "provenanceId": "dc/base/Schema",
                    }
                ]
            }
        )
        ft = read_five_tuple(arcs)
        assert ft.meas_qual_dcid is None

    def test_resolved_dcid_qualifier_is_unaffected(self):
        arcs = self._base_arcs(measurementQualifier={"nodes": [{"dcid": "Nominal"}]})
        ft = read_five_tuple(arcs)
        assert ft.meas_qual_dcid == "Nominal"


class TestDenominatorIdentityPrecedence:
    """Display and identity shapes prioritise unresolved state.

    This applies when the denominator also carries a dcid."""

    def _five_tuple(self, *, meas_denom_dcid: str | None, meas_denom_unresolved: bool) -> FiveTuple:
        return FiveTuple(
            pop_type_dcid="EconomicActivity",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=meas_denom_dcid,
            meas_denom_unresolved=meas_denom_unresolved,
        )

    def test_shape_id_unresolved_wins_over_a_co_present_dcid(self):
        ft = self._five_tuple(meas_denom_dcid="PerCapita", meas_denom_unresolved=True)
        assert standard_shape_id_for(ft).endswith("_per_unresolved")

    def test_shape_id_uses_the_dcid_when_not_flagged_unresolved(self):
        ft = self._five_tuple(meas_denom_dcid="Count_Person", meas_denom_unresolved=False)
        assert standard_shape_id_for(ft).endswith("_per_count_person")

    def test_shape_id_has_no_denominator_suffix_when_absent(self):
        ft = self._five_tuple(meas_denom_dcid=None, meas_denom_unresolved=False)
        assert not standard_shape_id_for(ft).endswith(("_per_unresolved", "_per_count_person"))

    def test_identity_shape_unresolved_wins_over_a_co_present_dcid(self):
        shape = identity_shape_for(
            pop_type_dcid="EconomicActivity",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid="PerCapita",
            meas_denom_unresolved=True,
        )
        assert shape["measurement_denominator"] == {"unresolved": True}

    def test_identity_shape_uses_the_dcid_when_not_flagged_unresolved(self):
        shape = identity_shape_for(
            pop_type_dcid="EconomicActivity",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid="Count_Person",
            meas_denom_unresolved=False,
        )
        assert shape["measurement_denominator"] == {"ref": "Count_Person"}


class TestReadConstraints:
    def test_three_constraints_read_from_arcs(self):
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        constraints = read_constraints(arcs)
        assert constraints[PROP_SCHEME] == "ODAGrants"
        assert constraints[PROP_PURPOSE] == "DAC/Health"
        assert constraints[PROP_RECIPIENT] == "country/ETH"

    def test_no_constraint_properties_returns_empty(self):
        arcs = {
            "populationType": {"nodes": [{"dcid": "SomePop"}]},
        }
        constraints = read_constraints(arcs)
        assert constraints == {}

    def test_constraint_with_missing_value_omitted(self):
        arcs = {
            "constraintProperties": {"nodes": [{"dcid": "SomeProp"}]},
        }
        constraints = read_constraints(arcs)
        assert "SomeProp" not in constraints


class TestReadUnresolvedConstraints:
    """Separates absent constraints from present unresolved values."""

    def test_resolved_property_not_in_unresolved_set(self):
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        assert read_unresolved_constraints(arcs) == frozenset()

    def test_absent_property_not_in_unresolved_set(self):
        arcs = {
            "constraintProperties": {"nodes": [{"dcid": PROP_RECIPIENT}]},
        }
        assert read_unresolved_constraints(arcs) == frozenset()

    def test_present_but_unresolved_property_is_flagged(self):
        arcs = {
            "constraintProperties": {"nodes": [{"dcid": PROP_RECIPIENT}]},
            PROP_RECIPIENT: {
                "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
            },
        }
        assert read_unresolved_constraints(arcs) == frozenset({PROP_RECIPIENT})
        # A present unresolved value belongs only to this set.
        assert PROP_RECIPIENT not in read_constraints(arcs)


class TestReadDeclaredRoles:
    def test_absent_arc_returns_observation_about(self):
        arcs = {"populationType": {"nodes": [{"dcid": "Person"}]}}
        assert read_declared_roles(arcs) == ("observationAbout",)

    def test_empty_arc_returns_observation_about(self):
        arcs = {"observationProperties": {"nodes": []}}
        assert read_declared_roles(arcs) == ("observationAbout",)

    def test_two_roles_returned_in_arc_order(self):
        arcs = {"observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]}}
        assert read_declared_roles(arcs) == ("provider", "recipient")

    def test_lone_observation_about_raises_degenerate(self):
        """A lone ``observationAbout`` role raises ``DegenerateObservationProperties``."""
        arcs = {"observationProperties": {"nodes": [{"dcid": "observationAbout"}]}}
        with pytest.raises(DegenerateObservationProperties):
            read_declared_roles(arcs)

    def test_observation_about_alongside_another_role_raises_degenerate(self):
        arcs = {
            "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "observationAbout"}]}
        }
        with pytest.raises(DegenerateObservationProperties):
            read_declared_roles(arcs)


class TestReadSlotTaxonomy:
    """Builds taxonomy from the supplied member facts."""

    def test_three_constraint_group_gets_observed_union(self):
        """The taxonomy contains the confirmed member's three constraint values."""
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)
        assert len(shapes) == 1
        shape = shapes[0]

        assert shape.slot_taxonomy is not None
        taxonomy = shape.slot_taxonomy

        assert taxonomy[f"how:{PROP_SCHEME}"] == ["ODAGrants"]
        assert taxonomy[f"how:{PROP_PURPOSE}"] == ["DAC/Health"]
        assert taxonomy[f"where:{PROP_RECIPIENT}"] == ["country/ETH"]

    def test_standard_shape_observed_union_taxonomy(self):
        """Matching members contribute both age values."""

        def _std_arcs(age_value: str) -> dict:
            return {
                "populationType": {"nodes": [{"dcid": "Person"}]},
                "measuredProperty": {"nodes": [{"dcid": "count"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "constraintProperties": {"nodes": [{"dcid": "age"}]},
                "age": {"nodes": [{"dcid": age_value}]},
            }

        sv1 = "Count_Person_15To64Years"
        sv2 = "Count_Person_Upto14Years"
        arcs1 = _std_arcs("Years15To64")
        arcs2 = _std_arcs("Upto14Years")

        nodes = {
            "age": {"label": "Age Group"},
            sv1: {"label": "Population 15-64", "arcs": arcs1},
            sv2: {"label": "Population 0-14", "arcs": arcs2},
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[sv1, sv2], graph=graph)

        assert len(shapes) == 1
        shape = shapes[0]

        taxonomy = shape.slot_taxonomy
        assert taxonomy is not None

        age_values_found = []
        for key, vals in taxonomy.items():
            if "age" in key:
                age_values_found = vals
                break

        assert "Years15To64" in age_values_found
        assert "Upto14Years" in age_values_found

    def test_read_slot_taxonomy_direct_three_constraint_shape(self):
        """The direct reader returns the three-constraint union."""
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)
        shape = shapes[0]

        taxonomy = read_slot_taxonomy(
            slot_keys=shape.slot_keys,
            member_facts={
                sv: ArcsMemberFacts(arcs=arcs) for sv, arcs in (shape.sv_arc_facts or {}).items()
            },
        )

        assert taxonomy[f"how:{PROP_SCHEME}"] == ["ODAGrants"]
        assert taxonomy[f"how:{PROP_PURPOSE}"] == ["DAC/Health"]
        assert taxonomy[f"where:{PROP_RECIPIENT}"] == ["country/ETH"]
