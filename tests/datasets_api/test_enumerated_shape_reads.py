"""Tests pin described-identity filtering and slot binding in `resolve_definition_members`,
including denominator, hierarchy, and stored-identity cases.
"""

from __future__ import annotations

from dfo_spec_kit.identity import spec_id_from_value

from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.bound_values import bound_values_from_bindings
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from statvar_members import UNRESOLVED_NODE, MemberEnumerator, slot_bindings_from_identity

from ._graph_doubles import FakeMintGraphReader
from .conftest import FakeDatasetStore

_SPEC_ID = "spec_test"


def _resolve(identity: dict, *, graph: FakeMintGraphReader):
    return resolve_definition_members(
        identity,
        spec_id=_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=get_index(),
    ).shapes


def _identity(*, measured_property: str, population_type: str, slots: list[dict]) -> dict:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": population_type,
            "measured_property": measured_property,
        },
        "slots": slots,
        "entities": {},
        "requested_window": {"kind": "all"},
    }


_GENDER_FAMILY_ARCS = {
    "Count_Person": {
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "observationProperties": (),
    },
    "Count_Person_Male": {
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "constraintProperties": ("gender",),
        "gender": ("Male",),
        "observationProperties": (),
    },
    "Count_Person_Female": {
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "constraintProperties": ("gender",),
        "gender": ("Female",),
        "observationProperties": (),
    },
}
"""One unconstrained member and two gender-constrained members."""


def _gender_family_graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            dcid: {"typeOf": ("StatisticalVariable",), **arcs}
            for dcid, arcs in _GENDER_FAMILY_ARCS.items()
        }
    )


def test_unbound_slot_is_answered_by_the_single_least_narrowed_member():
    identity = _identity(
        measured_property="count",
        population_type="Person",
        slots=[{"property": "gender", "binding": {"state": "unbound"}}],
    )

    shapes = _resolve(identity, graph=_gender_family_graph())

    assert [shape.stat_var for shape in shapes] == ["dcid:Count_Person"]
    assert shapes[0].slots == ()


def test_value_binding_narrows_to_exactly_the_member_asserting_it():
    identity = _identity(
        measured_property="count",
        population_type="Person",
        slots=[{"property": "gender", "binding": {"state": "value", "ref": "Male"}}],
    )

    shapes = _resolve(identity, graph=_gender_family_graph())

    assert [shape.stat_var for shape in shapes] == ["dcid:Count_Person_Male"]
    assert shapes[0].slots == (("gender", "Male"),)


def test_set_binding_admits_every_named_value_and_excludes_the_rest():
    identity = _identity(
        measured_property="count",
        population_type="Person",
        slots=[{"property": "gender", "binding": {"state": "set", "refs": ["Male", "Female"]}}],
    )

    shapes = _resolve(identity, graph=_gender_family_graph())

    assert sorted(shape.stat_var for shape in shapes) == [
        "dcid:Count_Person_Female",
        "dcid:Count_Person_Male",
    ]


def test_no_dominating_member_keeps_both_narrowed_siblings():
    # Mirrors `tests.statvar_members.conftest._MALE_FEMALE_ONLY_ARCS`: no
    # unconstrained `Count_Person` node exists on this graph, so the
    # antichain has no subset relation to reduce over and keeps both.
    graph = FakeMintGraphReader(
        arcs={
            dcid: {"typeOf": ("StatisticalVariable",), **arcs}
            for dcid, arcs in _GENDER_FAMILY_ARCS.items()
            if dcid != "Count_Person"
        }
    )
    identity = _identity(
        measured_property="count",
        population_type="Person",
        slots=[{"property": "gender", "binding": {"state": "unbound"}}],
    )

    shapes = _resolve(identity, graph=graph)

    assert sorted(shape.stat_var for shape in shapes) == [
        "dcid:Count_Person_Female",
        "dcid:Count_Person_Male",
    ]


_DENOMINATOR_ARCS = {
    "GdpPerCapita_Resolved": {
        "populationType": ("Country",),
        "measuredProperty": ("amount",),
        "measurementDenominator": ("Count_Person",),
        "observationProperties": (),
    },
    "GdpPerCapita_Unresolved": {
        "populationType": ("Country",),
        "measuredProperty": ("amount",),
        "measurementDenominator": (UNRESOLVED_NODE,),
        "observationProperties": (),
    },
    "Gdp_NoDenominator": {
        "populationType": ("Country",),
        "measuredProperty": ("amount",),
        "observationProperties": (),
    },
}


def _denominator_graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            dcid: {"typeOf": ("StatisticalVariable",), **arcs}
            for dcid, arcs in _DENOMINATOR_ARCS.items()
        }
    )


def test_measurement_denominator_ref_selects_only_the_matching_resolved_member():
    identity = _identity(measured_property="amount", population_type="Country", slots=[])
    identity["shape"]["measurement_denominator"] = {"ref": "Count_Person"}

    shapes = _resolve(identity, graph=_denominator_graph())

    assert [shape.stat_var for shape in shapes] == ["dcid:GdpPerCapita_Resolved"]


def test_measurement_denominator_unresolved_selects_only_the_unresolvable_arc_member():
    # `denominator_unresolved=True` means "a denominator arc exists whose
    # target node carries no dcid". The reader surfaces such targets as
    # `UNRESOLVED_NODE`, so a member whose denominator resolves to a real
    # dcid carries a different identity and is excluded.
    identity = _identity(measured_property="amount", population_type="Country", slots=[])
    identity["shape"]["measurement_denominator"] = {"unresolved": True}

    shapes = _resolve(identity, graph=_denominator_graph())

    assert [shape.stat_var for shape in shapes] == ["dcid:GdpPerCapita_Unresolved"]


def test_absent_measurement_denominator_selects_only_the_member_with_no_denominator_arc():
    identity = _identity(measured_property="amount", population_type="Country", slots=[])

    shapes = _resolve(identity, graph=_denominator_graph())

    assert [shape.stat_var for shape in shapes] == ["dcid:Gdp_NoDenominator"]


def test_measurement_denominator_ref_resolves_against_a_large_denominator_less_family():
    """A denominator query remains usable with 40 denominator-less siblings, above the
    32-candidate reprobe budget, because the graph filter excludes them."""
    graph = FakeMintGraphReader(
        arcs={
            "GdpPerCapita_Resolved": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("Country",),
                "measuredProperty": ("amount",),
                "measurementDenominator": ("Count_Person",),
                "observationProperties": (),
            },
            **{
                f"Gdp_NoDenom{i:02d}": {
                    "typeOf": ("StatisticalVariable",),
                    "populationType": ("Country",),
                    "measuredProperty": ("amount",),
                    "observationProperties": (),
                }
                for i in range(40)
            },
        }
    )
    identity = _identity(measured_property="amount", population_type="Country", slots=[])
    identity["shape"]["measurement_denominator"] = {"ref": "Count_Person"}

    shapes = _resolve(identity, graph=graph)

    assert [shape.stat_var for shape in shapes] == ["dcid:GdpPerCapita_Resolved"]


def test_stat_type_and_measurement_qualifier_narrow_the_ask():
    graph = FakeMintGraphReader(
        arcs={
            "Amount_Projected": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "statType": ("projectedValue",),
                "measurementQualifier": ("commitment",),
                "observationProperties": (),
            },
            "Amount_Measured": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "statType": ("measuredValue",),
                "observationProperties": (),
            },
        }
    )
    identity = {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "DevelopmentFinance",
            "measured_property": "amount",
            "stat_type": "projectedValue",
            "measurement_qualifier": "commitment",
        },
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }

    shapes = _resolve(identity, graph=graph)

    assert [shape.stat_var for shape in shapes] == ["dcid:Amount_Projected"]


def test_registry_storage_keeps_an_unbound_slot_through_the_round_trip():
    """The stored identity must retain an unbound slot's property so enumeration uses the
    same vocabulary as the original ask. With no unconstrained member, both identities resolve
    to the same two gender-specific members."""
    male_female_graph = FakeMintGraphReader(
        arcs={
            dcid: {"typeOf": ("StatisticalVariable",), **arcs}
            for dcid, arcs in _GENDER_FAMILY_ARCS.items()
            if dcid != "Count_Person"
        }
    )
    fresh_identity = _identity(
        measured_property="count",
        population_type="Person",
        slots=[{"property": "gender", "binding": {"state": "unbound"}}],
    )

    # The fresh (dense) ask resolves both narrowed siblings -- no
    # unconstrained member exists to dominate them.
    fresh_shapes = _resolve(fresh_identity, graph=male_female_graph)
    assert sorted(shape.stat_var for shape in fresh_shapes) == [
        "dcid:Count_Person_Female",
        "dcid:Count_Person_Male",
    ]

    spec_id = recompute_identity(fresh_identity).spec_id
    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    registry.register_definition(
        identity_definition=fresh_identity, schema_version=1, supplied_spec_id=spec_id
    )
    stored = LiveSpecReader(store=store).read_definition(spec_id)
    assert stored is not None
    assert stored.identity["slots"] == [{"property": "gender", "binding": {"state": "unbound"}}]

    stored_shapes = resolve_definition_members(
        stored.identity,
        spec_id=spec_id,
        graph=male_female_graph,
        enumerator=MemberEnumerator(graph=male_female_graph),
        code_index=get_index(),
    ).shapes
    assert sorted(shape.stat_var for shape in stored_shapes) == [
        "dcid:Count_Person_Female",
        "dcid:Count_Person_Male",
    ]


# Hierarchical value bindings expand to descendant codes, regardless of parent row counts.


def test_hierarchical_parent_value_binding_widens_to_its_real_leaf_member():
    """A value binding to hollow parent `CRSTypeOfFinance100` resolves its descendant
    `CRSTypeOfFinance110`, and the returned shape carries the leaf value."""
    identity = _identity(
        measured_property="amount",
        population_type="DevelopmentFinance",
        slots=[
            {
                "property": "crsTypeOfFinance",
                "binding": {"state": "value", "ref": "CRSTypeOfFinance100"},
            }
        ],
    )
    graph = FakeMintGraphReader(
        arcs={
            "OdaGrantsToKenya": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": (),
                "constraintProperties": ("crsTypeOfFinance",),
                "crsTypeOfFinance": ("CRSTypeOfFinance110",),
            }
        }
    )

    shapes = _resolve(identity, graph=graph)

    assert [shape.stat_var for shape in shapes] == ["dcid:OdaGrantsToKenya"]
    assert shapes[0].slots == (("crsTypeOfFinance", "CRSTypeOfFinance110"),)


def test_flat_code_system_value_binding_is_unaffected_by_widening():
    """A flat `crsTypeOfFlow` value binding remains exact. The sibling code is excluded."""
    identity = _identity(
        measured_property="amount",
        population_type="DevelopmentFinance",
        slots=[
            {"property": "crsTypeOfFlow", "binding": {"state": "value", "ref": "CRSTypeOfFlow10"}}
        ],
    )
    graph = FakeMintGraphReader(
        arcs={
            "OdaGrant": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": (),
                "constraintProperties": ("crsTypeOfFlow",),
                "crsTypeOfFlow": ("CRSTypeOfFlow10",),
            },
            "OdaOtherFlow": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": (),
                "constraintProperties": ("crsTypeOfFlow",),
                "crsTypeOfFlow": ("CRSTypeOfFlow21",),
            },
        }
    )

    shapes = _resolve(identity, graph=graph)

    assert [shape.stat_var for shape in shapes] == ["dcid:OdaGrant"]


def test_an_exact_hierarchical_parent_binding_exports_only_its_own_rows():
    """An `exact` binding selects only the parent code. The index has 12 direct StatVar rows
    and 8,152 rows across 99 descendants. Exact selection excludes those descendants."""
    identity = _identity(
        measured_property="amount",
        population_type="DevelopmentFinance",
        slots=[
            {"property": "crsPurpose", "binding": {"state": "exact", "ref": "CRSPurposeCode100"}}
        ],
    )
    graph = FakeMintGraphReader(
        arcs={
            "OdaGeneralToKenya": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": (),
                "constraintProperties": ("crsPurpose",),
                "crsPurpose": ("CRSPurposeCode100",),
            },
            "OdaEducationToKenya": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": (),
                "constraintProperties": ("crsPurpose",),
                # Exact selection excludes this descendant, regardless of hollowness.
                "crsPurpose": ("CRSPurposeCode11110",),
            },
        }
    )

    shapes = _resolve(identity, graph=graph)

    assert [shape.stat_var for shape in shapes] == ["dcid:OdaGeneralToKenya"]
    assert shapes[0].slots == (("crsPurpose", "CRSPurposeCode100"),)


def test_widening_agrees_with_searchs_own_descendant_expansion_for_a_hollow_parent():
    """Search and export enumeration must widen hollow parent `CRSTypeOfFinance100` to the
    same descendant set."""
    binding = SlotBindingDraft(
        axis="how",
        property_dcid="crsTypeOfFinance",
        kind="value",
        value_dcids=["CRSTypeOfFinance100"],
    )
    searchs_widened = bound_values_from_bindings([binding])

    exporters_widened = slot_bindings_from_identity(
        [
            {
                "property": "crsTypeOfFinance",
                "binding": {"state": "value", "ref": "CRSTypeOfFinance100"},
            }
        ],
        widen=get_index().descendants,
    )

    assert exporters_widened == {
        "crsTypeOfFinance": ("set", tuple(searchs_widened["crsTypeOfFinance"]))
    }
    # Not a vacuous check: the parent genuinely widened to more than itself.
    assert len(searchs_widened["crsTypeOfFinance"]) > 1


def test_widening_agrees_with_searchs_own_descendant_expansion_for_a_non_hollow_parent():
    """Search and export enumeration widen non-hollow `CRSPurposeCode100` to the same
    subtree, which has 12 direct and 8,152 total StatVar rows."""
    binding = SlotBindingDraft(
        axis="how",
        property_dcid="crsPurpose",
        kind="value",
        value_dcids=["CRSPurposeCode100"],
    )
    searchs_widened = bound_values_from_bindings([binding])
    # Sanity: retrieval genuinely widened past the bare value.
    assert len(searchs_widened["crsPurpose"]) > 1

    exporters_widened = slot_bindings_from_identity(
        [{"property": "crsPurpose", "binding": {"state": "value", "ref": "CRSPurposeCode100"}}],
        widen=get_index().descendants,
    )

    assert exporters_widened == {"crsPurpose": ("set", tuple(searchs_widened["crsPurpose"]))}


def test_widening_agrees_with_searchs_own_descendant_expansion_for_a_hollow_set_binding():
    """Search and export enumeration must expand a set binding to the same descendants."""
    binding = SlotBindingDraft(
        axis="how",
        property_dcid="crsTypeOfFinance",
        kind="set",
        value_dcids=["CRSTypeOfFinance100"],
    )
    searchs_widened = bound_values_from_bindings([binding])

    exporters_widened = slot_bindings_from_identity(
        [
            {
                "property": "crsTypeOfFinance",
                "binding": {"state": "set", "refs": ["CRSTypeOfFinance100"]},
            }
        ],
        widen=get_index().descendants,
    )

    assert exporters_widened == {
        "crsTypeOfFinance": ("set", tuple(searchs_widened["crsTypeOfFinance"]))
    }
    assert len(searchs_widened["crsTypeOfFinance"]) > 1


# Both hierarchical slots widen to their descendant sets before member resolution.


def test_two_hierarchical_category_slots_both_widen_and_the_leaf_member_resolves():
    """The ask resolves a member asserting both descendant codes after widening its two
    hierarchical slots."""
    identity = _identity(
        measured_property="amount",
        population_type="DevelopmentFinance",
        slots=[
            {"property": "crsPurpose", "binding": {"state": "value", "ref": "CRSPurposeCode110"}},
            {
                "property": "crsTypeOfFinance",
                "binding": {"state": "value", "ref": "CRSTypeOfFinance100"},
            },
        ],
    )
    graph = FakeMintGraphReader(
        arcs={
            "OdaEducationGrantToNigeria": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": (),
                "constraintProperties": ("crsPurpose", "crsTypeOfFinance"),
                "crsPurpose": ("CRSPurposeCode11220",),
                "crsTypeOfFinance": ("CRSTypeOfFinance110",),
            }
        }
    )
    # Sanity: this is genuinely the non-hollow case the old gate turned on,
    # not an artifact of a stale packaged index.
    assert len(get_index().descendants("CRSPurposeCode110")) > 1

    shapes = _resolve(identity, graph=graph)

    assert [shape.stat_var for shape in shapes] == ["dcid:OdaEducationGrantToNigeria"]
    assert shapes[0].slots == (
        ("crsPurpose", "CRSPurposeCode11220"),
        ("crsTypeOfFinance", "CRSTypeOfFinance110"),
    )


def test_identity_hash_unchanged_by_resolving_a_two_category_slot_spec():
    """Member resolution must leave the identity unchanged because widening operates on a
    translated copy."""
    identity = _identity(
        measured_property="amount",
        population_type="DevelopmentFinance",
        slots=[
            {"property": "crsPurpose", "binding": {"state": "value", "ref": "CRSPurposeCode110"}},
            {
                "property": "crsTypeOfFinance",
                "binding": {"state": "value", "ref": "CRSTypeOfFinance100"},
            },
        ],
    )
    graph = FakeMintGraphReader(
        arcs={
            "OdaEducationGrantToNigeria": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": (),
                "constraintProperties": ("crsPurpose", "crsTypeOfFinance"),
                "crsPurpose": ("CRSPurposeCode11220",),
                "crsTypeOfFinance": ("CRSTypeOfFinance110",),
            }
        }
    )

    before = spec_id_from_value(identity)
    _resolve(identity, graph=graph)
    after = spec_id_from_value(identity)

    assert before == after
