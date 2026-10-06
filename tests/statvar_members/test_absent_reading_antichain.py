"""Pins antichain reduction by absent-reading metadata, including the default `aggregate` rule."""

from __future__ import annotations

import inspect

from statvar_members import dominates_constraint_keys
from statvar_members.cache import MemberEnumerator
from statvar_members.enumerate import enumerate_members
from statvar_members.types import EMPTY_READINGS, FiveTuple

from .conftest import FakeMemberArcReader, kenya_reader

_FIVE_TUPLE = FiveTuple(measured_property="amount")
_SLOTS = {"purpose": ("unbound",), "modality": ("unbound",)}


# ---------------------------------------------------------------------------
# dominates_constraint_keys: pure domination-rule unit tests, independent of
# a live enumeration. search.engine.retrieval.read_plan.minimal_constraint_svs
# reduces a preview's members by this predicate.
# ---------------------------------------------------------------------------


def test_dominates_requires_a_proper_subset():
    assert not dominates_constraint_keys(
        frozenset({"purpose"}), frozenset({"purpose"}), readings={}
    )
    assert not dominates_constraint_keys(
        frozenset({"purpose", "modality"}), frozenset({"purpose"}), readings={}
    )
    assert dominates_constraint_keys(frozenset(), frozenset({"purpose"}), readings={})


def test_dominates_refuses_a_not_declared_property():
    # An "aggregate" reading (the default when readings says nothing) lets
    # the dominator's silence on "modality" stand in for the dominated
    # member's declared value; "not_declared" refuses that -- the
    # dominator's silence is a vintage that never collected the property,
    # not a summary over it.
    assert dominates_constraint_keys(
        frozenset(), frozenset({"modality"}), readings={"modality": "aggregate"}
    )
    assert not dominates_constraint_keys(
        frozenset(), frozenset({"modality"}), readings={"modality": "not_declared"}
    )


def _purpose_modality_reader() -> FakeMemberArcReader:
    """`Flow_Purpose` has `{purpose}`; its sibling adds `modality` with the same purpose.
    The reading for absent `modality` decides whether the first dominates."""
    return FakeMemberArcReader(
        arcs={
            "Flow_Purpose": {
                "typeOf": ("StatisticalVariable",),
                "populationType": (),
                "measuredProperty": ("amount",),
                "statType": (),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("purpose",),
                "observationProperties": (),
                "purpose": ("Health",),
            },
            "Flow_PurposeModality": {
                "typeOf": ("StatisticalVariable",),
                "populationType": (),
                "measuredProperty": ("amount",),
                "statType": (),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("purpose", "modality"),
                "observationProperties": (),
                "purpose": ("Health",),
                "modality": ("Grant",),
            },
        }
    )


def test_not_declared_modality_keeps_both_members():
    """A `not_declared` modality separates the members, so both survive."""
    reader = _purpose_modality_reader()

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_SLOTS,
        graph=reader,
        readings={"modality": "not_declared"},
    )

    assert {member.stat_var for member in group.members} == {
        "Flow_Purpose",
        "Flow_PurposeModality",
    }


def test_aggregate_modality_keeps_only_the_least_constrained_member():
    """An `aggregate` modality lets the less-constrained member dominate."""
    reader = _purpose_modality_reader()

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots=_SLOTS,
        graph=reader,
        readings={"modality": "aggregate"},
    )

    assert {member.stat_var for member in group.members} == {"Flow_Purpose"}


def test_a_property_absent_from_readings_defaults_to_aggregate():
    """`readings` says nothing about `modality` at all: the default reading
    is `"aggregate"`, identical in outcome to stating it explicitly.
    """
    reader = _purpose_modality_reader()

    group = enumerate_members(five_tuple=_FIVE_TUPLE, slots=_SLOTS, graph=reader, readings={})

    assert {member.stat_var for member in group.members} == {"Flow_Purpose"}


def test_omitting_readings_entirely_behaves_exactly_like_an_empty_mapping():
    """Omitting `readings` uses `EMPTY_READINGS` and matches an explicit empty mapping."""
    reader = _purpose_modality_reader()

    omitted = enumerate_members(five_tuple=_FIVE_TUPLE, slots=_SLOTS, graph=reader)
    explicit_empty = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots=_SLOTS, graph=reader, readings={}
    )

    assert {member.stat_var for member in omitted.members} == {"Flow_Purpose"}
    assert omitted.members == explicit_empty.members


def test_readings_default_is_the_shared_immutable_empty_mapping():
    """Calls omitting `readings` share the same immutable `EMPTY_READINGS` object."""
    assert inspect.signature(enumerate_members).parameters["readings"].default is EMPTY_READINGS


def test_demographic_reduction_is_byte_for_byte_unchanged_with_no_readings():
    """With no absence rule, `Person` still dominates its male and female members."""
    reader = kenya_reader()

    group = enumerate_members(
        five_tuple=FiveTuple(
            population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
        ),
        slots={"gender": ("unbound",)},
        graph=reader,
        readings={},
    )

    assert [member.stat_var for member in group.members] == ["LifeExpectancy_Person"]


def test_synthetic_system_opposite_of_the_real_crs_declarations_still_dominates():
    """Supplied readings govern reduction even for CRS-named properties.
    All four use `aggregate`."""
    reader = FakeMemberArcReader(
        arcs={
            "Flow_Purpose": {
                "typeOf": ("StatisticalVariable",),
                "populationType": (),
                "measuredProperty": ("amount",),
                "statType": (),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("crsPurpose",),
                "observationProperties": (),
                "crsPurpose": ("Health",),
            },
            "Flow_PurposeModality": {
                "typeOf": ("StatisticalVariable",),
                "populationType": (),
                "measuredProperty": ("amount",),
                "statType": (),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("crsPurpose", "crsCooperationModality"),
                "observationProperties": (),
                "crsPurpose": ("Health",),
                "crsCooperationModality": ("Grant",),
            },
        }
    )
    opposite_of_real_crs_declarations = {
        "crsPurpose": "aggregate",
        "crsTypeOfFinance": "aggregate",
        "crsTypeOfFlow": "aggregate",
        "crsCooperationModality": "aggregate",
    }

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE,
        slots={"crsPurpose": ("unbound",), "crsCooperationModality": ("unbound",)},
        graph=reader,
        readings=opposite_of_real_crs_declarations,
    )

    assert {member.stat_var for member in group.members} == {"Flow_Purpose"}


def test_cache_serves_a_different_group_for_the_same_key_under_different_readings():
    """Different `readings` values need separate cache entries because they change reduction."""
    reader = _purpose_modality_reader()
    enumerator = MemberEnumerator()

    not_declared = enumerator.enumerate(
        graph=reader,
        stamp_key="rev1",
        five_tuple=_FIVE_TUPLE,
        slots=_SLOTS,
        readings={"modality": "not_declared"},
    )
    aggregate = enumerator.enumerate(
        graph=reader,
        stamp_key="rev1",
        five_tuple=_FIVE_TUPLE,
        slots=_SLOTS,
        readings={"modality": "aggregate"},
    )

    assert {member.stat_var for member in not_declared.members} == {
        "Flow_Purpose",
        "Flow_PurposeModality",
    }
    assert {member.stat_var for member in aggregate.members} == {"Flow_Purpose"}
