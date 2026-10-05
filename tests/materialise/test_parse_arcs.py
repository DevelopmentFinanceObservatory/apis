"""Tests `parse_arcs` deduplication, ordering, unresolved-node handling, and missing properties."""

from __future__ import annotations

from materialise.graph import UNRESOLVED_NODE, parse_arcs


def test_dedups_a_value_declared_by_two_provenances():
    # The graph reports one arc entry per asserting provenance. A StatVar
    # shape property declared by two provenances with the same value comes
    # back as two entries carrying an identical dcid.
    arcs = {
        "measuredProperty": {
            "nodes": [
                {"dcid": "lifeExpectancy", "provenanceId": "dc/base/India_LifeExpectancy"},
                {"dcid": "lifeExpectancy", "provenanceId": "dc/base/Schema"},
            ]
        }
    }

    result = parse_arcs(arcs, ["measuredProperty"])

    assert result == {"measuredProperty": ("lifeExpectancy",)}


def test_preserves_first_occurrence_order_of_distinct_values():
    arcs = {
        "constraintProperties": {"nodes": [{"dcid": "gender"}, {"dcid": "age"}, {"dcid": "gender"}]}
    }

    result = parse_arcs(arcs, ["constraintProperties"])

    assert result == {"constraintProperties": ("gender", "age")}


def test_genuinely_conflicting_values_both_survive():
    # Dedup only ever collapses identical values. Two distinct values on
    # the same property are a real disagreement and both survive.
    arcs = {"gender": {"nodes": [{"dcid": "Male"}, {"dcid": "Female"}]}}

    result = parse_arcs(arcs, ["gender"])

    assert result == {"gender": ("Male", "Female")}


def test_unresolved_node_dedups_like_any_other_value():
    # Duplicate unresolved targets collapse to one presence marker.
    arcs = {"measurementDenominator": {"nodes": [{}, {}]}}

    result = parse_arcs(arcs, ["measurementDenominator"], mark_unresolved=True)

    assert result == {"measurementDenominator": (UNRESOLVED_NODE,)}


def test_unresolved_node_stays_distinct_from_a_resolved_value():
    arcs = {"measurementDenominator": {"nodes": [{"dcid": "Count_Person"}, {}]}}

    result = parse_arcs(arcs, ["measurementDenominator"], mark_unresolved=True)

    assert result == {"measurementDenominator": ("Count_Person", UNRESOLVED_NODE)}


def test_missing_property_maps_to_empty_tuple_not_omitted():
    result = parse_arcs({}, ["absent"])

    assert result == {"absent": ()}
