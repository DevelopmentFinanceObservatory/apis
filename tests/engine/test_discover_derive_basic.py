"""Tests shape grouping, labels, denominator states, and confirm limits."""

from __future__ import annotations

from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import STANDARD_SHAPE_LABEL
from tests.engine._discover_fixtures import (
    _TEST_SV_NAMESPACE,
    MEAS_DENOM_DCID,
    MEAS_PROP_DCID,
    MEAS_QUAL_DCID,
    POP_TYPE_DCID,
    PROP_PURPOSE,
    PROP_RECIPIENT,
    PROP_SCHEME,
    STAT_TYPE_DCID,
    _fake_graph_with_svs,
    _three_constraint_arcs,
)
from tests.fixtures import FakeGraph


class TestDeriveShapesThreeConstraintGroup:
    """Three-constraint groups use the standard shape path."""

    def test_single_sv_yields_one_shape(self):
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert len(shapes) == 1
        shape = shapes[0]
        assert shape.pop_type_dcid == POP_TYPE_DCID
        assert shape.meas_prop_dcid == MEAS_PROP_DCID
        assert shape.stat_type_dcid == STAT_TYPE_DCID
        assert shape.meas_qual_dcid == MEAS_QUAL_DCID
        assert shape.meas_denom_dcid == MEAS_DENOM_DCID

    def test_three_constraint_slots_on_correct_axes(self):
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)
        shape = shapes[0]

        constraint_slots = [s for s in shape.slot_keys if s.property_dcid is not None]
        assert len(constraint_slots) == 3

        # PROP_RECIPIENT's value is a country/ dcid, so classify_axis's
        # place-fraction auto-rule sends it to "where". PROP_SCHEME and
        # PROP_PURPOSE take neither a place- nor a date-shaped value, so both
        # fall through to the "how" default.
        slot_by_prop = {s.property_dcid: s for s in constraint_slots}
        assert slot_by_prop[PROP_SCHEME].axis == "how"
        assert slot_by_prop[PROP_PURPOSE].axis == "how"
        assert slot_by_prop[PROP_RECIPIENT].axis == "where"

    def test_when_and_source_slots_appended(self):
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)
        shape = shapes[0]

        axes = [s.axis for s in shape.slot_keys]
        assert "when" in axes
        assert "source" in axes

    def test_shape_carries_sv_arc_facts(self):
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)
        shape = shapes[0]

        assert shape.sv_arc_facts is not None
        assert sv_dcid in shape.sv_arc_facts

    def test_shape_id_is_the_five_tuple_slug(self):
        """The shape id is the five-tuple slug."""
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert shapes[0].shape_id == f"{POP_TYPE_DCID}_{MEAS_PROP_DCID}_{STAT_TYPE_DCID}".lower()

    def test_multiple_svs_same_five_tuple_one_shape(self):
        """Matching five-tuples form one shape."""
        sv1 = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        sv2 = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-KEN"
        arcs1 = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        arcs2 = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/KEN")
        graph = _fake_graph_with_svs({sv1: arcs1, sv2: arcs2})

        shapes = derive_shapes(confirmed_svs=[sv1, sv2], graph=graph)

        assert len(shapes) == 1

    def test_label_sourced_from_name_arc_like_any_standard_shape(self):
        """The representative member's ``name`` arc supplies the label."""
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        arcs["name"] = {"nodes": [{"value": "Test fixture Health ODA Grants ETH"}]}
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert shapes[0].label == "Test fixture Health ODA Grants ETH"


class TestDeriveShapesPerConceptLabel:
    """Labels come from the representative member or the standard fallback."""

    def test_standard_shape_uses_representative_name_arc(self):
        sv_dcid = "Count_Person_Test"
        arcs = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "name": {"nodes": [{"value": "Total Population"}]},
        }
        nodes = {sv_dcid: {"label": "Total Population", "arcs": arcs}}
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert len(shapes) == 1
        assert shapes[0].label == "Total Population"

    def test_nameless_group_takes_the_standard_shape_label(self):
        """A nameless group uses ``STANDARD_SHAPE_LABEL``."""
        sv_dcid = "Count_Person_NoName"
        arcs = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        nodes = {sv_dcid: {"label": "whatever", "arcs": arcs}}
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert len(shapes) == 1
        assert shapes[0].label == STANDARD_SHAPE_LABEL

    def test_representative_sv_is_first_in_group_when_multiple_present(self):
        """The highest-scoring member supplies the group label."""
        sv1 = "Count_Person_First"
        sv2 = "Count_Person_Second"

        def _arcs(name: str) -> dict:
            return {
                "populationType": {"nodes": [{"dcid": "Person"}]},
                "measuredProperty": {"nodes": [{"dcid": "count"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "name": {"nodes": [{"value": name}]},
            }

        nodes = {
            sv1: {"label": "First", "arcs": _arcs("First Concept Name")},
            sv2: {"label": "Second", "arcs": _arcs("Second Concept Name")},
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(
            confirmed_svs=[sv1, sv2],
            graph=graph,
            sv_scores={sv1: 0.9, sv2: 0.5},
        )

        assert len(shapes) == 1
        assert shapes[0].label == "First Concept Name"


class TestDeriveShapesMeasDenomTriState:
    """Absent, unresolved, and resolved denominators produce distinct shape ids."""

    def _gdp_arcs(self, *, denominator: dict | None) -> dict:
        arcs = {
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        if denominator is not None:
            arcs["measurementDenominator"] = denominator
        return arcs

    def test_absent_and_unresolved_denominator_land_in_distinct_shapes(self):
        bare_sv = "Amount_EconomicActivity_Bare"
        unresolved_sv = "Amount_EconomicActivity_Unresolved"
        nodes = {
            bare_sv: {"label": "bare", "arcs": self._gdp_arcs(denominator=None)},
            unresolved_sv: {
                "label": "unresolved",
                "arcs": self._gdp_arcs(
                    denominator={
                        "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
                    }
                ),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[bare_sv, unresolved_sv], graph=graph)

        assert len(shapes) == 2
        shape_ids = {s.shape_id for s in shapes}
        assert len(shape_ids) == 2, f"expected distinct shape_ids, got {shape_ids}"
        bare_shape = next(s for s in shapes if not s.meas_denom_unresolved)
        unresolved_shape = next(s for s in shapes if s.meas_denom_unresolved)
        assert not bare_shape.shape_id.endswith("_unresolved")
        assert unresolved_shape.shape_id.endswith("_per_unresolved")

    def test_unresolved_shape_id_is_stable_and_distinct_from_resolved(self):
        unresolved_sv = "Amount_EconomicActivity_Unresolved2"
        resolved_sv = "Amount_EconomicActivity_Resolved2"
        nodes = {
            unresolved_sv: {
                "label": "unresolved",
                "arcs": self._gdp_arcs(
                    denominator={
                        "nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]
                    }
                ),
            },
            resolved_sv: {
                "label": "resolved",
                "arcs": self._gdp_arcs(denominator={"nodes": [{"dcid": "Count_Person"}]}),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[unresolved_sv, resolved_sv], graph=graph)

        assert len(shapes) == 2
        shape_ids = {s.shape_id for s in shapes}
        assert len(shape_ids) == 2
        assert "economicactivity_amount_measuredvalue_per_unresolved" in shape_ids
        assert "economicactivity_amount_measuredvalue_per_count_person" in shape_ids

        shapes_again = derive_shapes(confirmed_svs=[unresolved_sv, resolved_sv], graph=graph)
        assert {s.shape_id for s in shapes_again} == shape_ids

    def test_dcid_bearing_provisional_denominator_still_groups_as_unresolved(self):
        """A dcid-bearing ``PerCapita`` denominator typed ``ProvisionalNode`` is unresolved.

        It stays apart from the resolved sibling, as the availability sibling pruning requires.
        """
        unresolved_sv = "Amount_EconomicActivity_Unresolved3"
        resolved_sv = "Amount_EconomicActivity_Resolved3"
        nodes = {
            unresolved_sv: {
                "label": "unresolved",
                "arcs": self._gdp_arcs(
                    denominator={
                        "nodes": [
                            {
                                "dcid": "PerCapita",
                                "types": ["ProvisionalNode"],
                                "provenanceId": "dc/base/Schema",
                            }
                        ]
                    }
                ),
            },
            resolved_sv: {
                "label": "resolved",
                "arcs": self._gdp_arcs(denominator={"nodes": [{"dcid": "Count_Person"}]}),
            },
        }
        graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[unresolved_sv, resolved_sv], graph=graph)

        assert len(shapes) == 2
        shape_ids = {s.shape_id for s in shapes}
        assert "economicactivity_amount_measuredvalue_per_unresolved" in shape_ids
        assert "economicactivity_amount_measuredvalue_per_count_person" in shape_ids
        provisional_shape = next(s for s in shapes if s.meas_denom_unresolved)
        assert provisional_shape.meas_denom_dcid == "PerCapita"


class TestDeriveShapesUnresolvedConstraintNoSlotKey:
    """An unresolved-only property supplies no slot key.

    Binding has no value for that property."""

    def test_only_unresolved_property_yields_no_slot_key(self):
        sv_dcid = "UnemploymentRate_Person_OnlyUnresolvedGender"
        arcs = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": [{"dcid": "gender"}]},
            "gender": {"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]},
        }
        graph = _fake_graph_with_svs({sv_dcid: arcs})

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)

        assert len(shapes) == 1
        shape = shapes[0]
        assert "gender" not in {slot.property_dcid for slot in shape.slot_keys}
        assert shape.slot_taxonomy == {}


class TestDeriveShapesDropsUnconfirmable:
    """Unconfirmable StatVar ids are dropped without fabricated graph references."""

    def test_unconfirmable_sv_dropped(self):
        real_sv = "Count_Person"
        fake_sv = "Count_Person_DoesNotExist"
        arcs = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        graph = _fake_graph_with_svs({real_sv: arcs})

        shapes = derive_shapes(confirmed_svs=[real_sv, fake_sv], graph=graph)

        assert len(shapes) == 1
        # The shape carries only the real SV's arc facts
        assert fake_sv not in (shapes[0].sv_arc_facts or {})
        assert real_sv in (shapes[0].sv_arc_facts or {})

    def test_all_unconfirmable_returns_empty(self):
        graph = _fake_graph_with_svs({})  # no SVs in nodes

        shapes = derive_shapes(
            confirmed_svs=["Count_Person_DoesNotExist"],
            graph=graph,
        )

        assert shapes == []

    def test_empty_candidate_list_returns_empty(self):
        graph = FakeGraph(nodes={}, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=[], graph=graph)

        assert shapes == []


class TestDeriveShapesC2OverFireGuard:
    """Confirmed members share one group per five-tuple."""

    def test_noisy_set_with_one_confirmed_sv_yields_one_shape(self):
        """One confirmed candidate yields one shape."""
        ns = _TEST_SV_NAMESPACE
        noisy_svs = [
            "dc/topic/sdg_3.b.2",
            f"{ns}Health-GrantScheme-Noise",
            f"{ns}BasicHealth-GrantScheme-Noise",
            f"{ns}Health-GrantScheme-ETH",  # this one confirms
        ]
        real_sv = f"{ns}Health-GrantScheme-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "GrantScheme", "country/ETH")
        graph = _fake_graph_with_svs({real_sv: arcs})

        shapes = derive_shapes(confirmed_svs=noisy_svs, graph=graph)

        assert len(shapes) == 1
        assert shapes[0].pop_type_dcid == POP_TYPE_DCID

    def test_all_noisy_svs_unconfirmable_yields_zero_shapes(self):
        """No confirmed candidates yields no shapes."""
        ns = _TEST_SV_NAMESPACE
        noisy_svs = [
            "dc/topic/sdg_3.b.2",
            f"{ns}Health-GrantScheme-Noise",
            f"{ns}BasicHealth-GrantScheme-Noise",
        ]
        graph = FakeGraph(nodes={}, obs={}, detect={}, resolve={})

        shapes = derive_shapes(confirmed_svs=noisy_svs, graph=graph)

        assert len(shapes) <= 1

    def test_recorded_svs_from_noisy_query_yield_one_group(self):
        """Matching recorded five-tuples form one group."""
        ns = _TEST_SV_NAMESPACE
        # Build multiple SVs, each varying its own constraint values, that
        # all share the one five-tuple under test.
        svs_and_arcs = {
            f"{ns}Health-GrantScheme-ETH": _three_constraint_arcs(
                "DAC/Health", "GrantScheme", "country/ETH"
            ),
            f"{ns}BasicHealth-GrantScheme-ETH": _three_constraint_arcs(
                "DAC/BasicHealth", "GrantScheme", "country/ETH"
            ),
            f"{ns}Health-ODAGrants-ETH": _three_constraint_arcs(
                "DAC/Health", "ODAGrants", "country/ETH"
            ),
        }
        graph = _fake_graph_with_svs(svs_and_arcs)

        shapes = derive_shapes(confirmed_svs=list(svs_and_arcs.keys()), graph=graph)

        # Every SV shares the same five-tuple, so exactly one group forms.
        assert len(shapes) == 1
        assert shapes[0].pop_type_dcid == POP_TYPE_DCID


class TestDeriveShapesValueLabelsFromCodeIndex:
    """Code-index labels omit graph suffixes that blocked query matching (#866)."""

    def test_code_index_value_gets_the_clean_label(self):
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-CRSFinance-ETH"
        crs_dcid = "CRSTypeOfFinance110"
        arcs = _three_constraint_arcs("DAC/Health", crs_dcid, "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})
        # CodeRow.label strips this parenthetical CRS suffix.
        graph._nodes[crs_dcid] = {"label": "standard grant (CRS Finance)"}

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)
        shape = shapes[0]

        assert shape.value_labels is not None
        assert shape.value_labels[crs_dcid] == "standard grant"

    def test_non_code_index_value_keeps_the_raw_graph_label(self):
        """Values absent from the index keep their graph label."""
        sv_dcid = f"{_TEST_SV_NAMESPACE}Health-ODAGrants-ETH"
        arcs = _three_constraint_arcs("DAC/Health", "ODAGrants", "country/ETH")
        graph = _fake_graph_with_svs({sv_dcid: arcs})
        graph._nodes["ODAGrants"] = {"label": "ODA grants"}

        shapes = derive_shapes(confirmed_svs=[sv_dcid], graph=graph)
        shape = shapes[0]

        assert shape.value_labels is not None
        assert shape.value_labels["ODAGrants"] == "ODA grants"
