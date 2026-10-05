"""Pins role defaults, constraint slots, refusals, and batched shape resolution."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from datasets_api.codes_materialise import MemberNotResolvable
from datasets_api.shape import resolve_shape, resolve_shapes
from materialise.graph import UNRESOLVED_NODE, Facet, FacetDefaultPolicy

from ._graph_doubles import FakeMintGraphReader

# A StatVar declaring two roles via `observationProperties`, plus one bound
# constraint slot.
_BILATERAL_ODA_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "observationProperties": ("provider", "recipient"),
    "measuredProperty": ("amount",),
    "populationType": ("DevelopmentFinance",),
    "constraintProperties": ("DevelopmentFinanceScheme",),
    "DevelopmentFinanceScheme": ("ODAGrants",),
}

# A single declared role, no constraint slots.
_RECIPIENT_ONLY_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "observationProperties": ("recipient",),
    "measuredProperty": ("amount",),
    "populationType": ("DevelopmentFinance",),
}

# `observationProperties` itself names `observationAbout`.
_DEGENERATE_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "observationProperties": ("observationAbout",),
    "measuredProperty": ("amount",),
    "populationType": ("DevelopmentFinance",),
}

# Modelled on the live-probed Count_Person/LifeExpectancy_Person shapes: no
# `observationProperties` arc at all, no `constraintProperties`.
_COUNT_PERSON_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "measuredProperty": ("count",),
    "populationType": ("Person",),
}

# Modelled on the live-probed Count_Person_Female shape: a single bound
# constraint slot, no declared roles beyond the default.
_COUNT_PERSON_FEMALE_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "measuredProperty": ("count",),
    "populationType": ("Person",),
    "constraintProperties": ("gender",),
    "gender": ("Female",),
}


def test_declares_roles_from_observation_properties_in_arc_order():
    graph = FakeMintGraphReader(arcs={"BilateralODAGrants": _BILATERAL_ODA_ARCS})

    shape = resolve_shape("dcid:BilateralODAGrants", graph=graph)

    assert shape.stat_var == "dcid:BilateralODAGrants"
    assert shape.declared_roles == ("provider", "recipient")
    assert shape.measured_property == "amount"
    assert shape.population_type == "DevelopmentFinance"
    assert shape.slots == (("DevelopmentFinanceScheme", "ODAGrants"),)


def test_declares_a_single_role_from_a_one_valued_observation_properties_arc():
    graph = FakeMintGraphReader(arcs={"RecipientOnly": _RECIPIENT_ONLY_ARCS})

    shape = resolve_shape("dcid:RecipientOnly", graph=graph)

    assert shape.declared_roles == ("recipient",)
    assert shape.slots == ()


def test_absent_observation_properties_arc_defaults_to_observation_about():
    graph = FakeMintGraphReader(arcs={"CountPerson": _COUNT_PERSON_ARCS})

    shape = resolve_shape("dcid:CountPerson", graph=graph)

    assert shape.declared_roles == ("observationAbout",)
    assert shape.measured_property == "count"
    assert shape.population_type == "Person"


def test_empty_observation_properties_arc_defaults_the_same_as_absent():
    # Absent and empty arcs both read as `()`.
    graph = FakeMintGraphReader(
        arcs={
            "CountPerson": {**_COUNT_PERSON_ARCS, "observationProperties": ()},
        }
    )

    shape = resolve_shape("dcid:CountPerson", graph=graph)

    assert shape.declared_roles == ("observationAbout",)


def test_resolves_constraint_slots_with_their_bound_values():
    graph = FakeMintGraphReader(arcs={"CountPersonFemale": _COUNT_PERSON_FEMALE_ARCS})

    shape = resolve_shape("dcid:CountPersonFemale", graph=graph)

    assert shape.declared_roles == ("observationAbout",)
    assert shape.slots == (("gender", "Female"),)


def test_a_constraint_property_the_statvar_leaves_unbound_is_none():
    graph = FakeMintGraphReader(
        arcs={
            "CountPersonFemale": {
                **_COUNT_PERSON_FEMALE_ARCS,
                "gender": (),  # constraintProperties names it, with no bound value
            }
        }
    )

    shape = resolve_shape("dcid:CountPersonFemale", graph=graph)

    assert shape.slots == (("gender", None),)


def test_a_multi_valued_constraint_arc_refuses_rather_than_pinning_either_value():
    # Multiple constraint values cannot identify one slot value.
    graph = FakeMintGraphReader(
        arcs={"CountPersonFemale": {**_COUNT_PERSON_FEMALE_ARCS, "gender": ("Female", "Male")}}
    )

    with pytest.raises(MemberNotResolvable) as excinfo:
        resolve_shape("dcid:CountPersonFemale", graph=graph)

    assert excinfo.value.members == ("dcid:CountPersonFemale",)
    assert "gender" in str(excinfo.value.__cause__)


def test_a_node_whose_typeof_omits_statisticalvariable_refuses():
    # The graph resolves this node as a `Class`.
    graph = FakeMintGraphReader(
        arcs={"DevelopmentFinance": {"typeOf": ("Class",), "name": ("Development Finance",)}}
    )

    with pytest.raises(MemberNotResolvable) as excinfo:
        resolve_shape("dcid:DevelopmentFinance", graph=graph)
    assert excinfo.value.members == ("dcid:DevelopmentFinance",)


def test_an_unheld_node_refuses_the_same_way_as_a_held_non_statvar():
    # An unheld dcid and a node with empty arcs both read as `()`, so they refuse identically.
    graph = FakeMintGraphReader(arcs={})

    with pytest.raises(MemberNotResolvable) as excinfo:
        resolve_shape("dcid:NotAKnownShape", graph=graph)
    assert excinfo.value.members == ("dcid:NotAKnownShape",)


def test_observation_about_declared_alongside_nothing_else_still_refuses():
    # The degenerate-role guard also rejects `observationAbout` as the only declared role.
    graph = FakeMintGraphReader(arcs={"Degenerate": _DEGENERATE_ARCS})

    with pytest.raises(MemberNotResolvable) as excinfo:
        resolve_shape("dcid:Degenerate", graph=graph)
    assert excinfo.value.members == ("dcid:Degenerate",)


def test_resolve_shapes_batches_two_constrained_statvars_into_one_second_call():
    # One second batch reads constraint values for both StatVars.
    graph = FakeMintGraphReader(
        arcs={
            "BilateralODAGrants": _BILATERAL_ODA_ARCS,
            "CountPersonFemale": _COUNT_PERSON_FEMALE_ARCS,
        }
    )

    shapes = resolve_shapes(("dcid:BilateralODAGrants", "dcid:CountPersonFemale"), graph=graph)

    assert shapes["dcid:BilateralODAGrants"] == resolve_shape(
        "dcid:BilateralODAGrants",
        graph=FakeMintGraphReader(arcs={"BilateralODAGrants": _BILATERAL_ODA_ARCS}),
    )
    assert shapes["dcid:CountPersonFemale"] == resolve_shape(
        "dcid:CountPersonFemale",
        graph=FakeMintGraphReader(arcs={"CountPersonFemale": _COUNT_PERSON_FEMALE_ARCS}),
    )
    assert graph.calls["node_arcs_many"] == 2
    assert graph.calls["node_arc_values"] == 0


def test_resolve_shapes_omits_a_failing_member_instead_of_raising():
    # Unresolvable members are omitted; only the valid StatVar remains.
    graph = FakeMintGraphReader(
        arcs={
            "CountPerson": _COUNT_PERSON_ARCS,
            "DevelopmentFinance": {"typeOf": ("Class",), "name": ("Development Finance",)},
            "CountPersonFemale": {**_COUNT_PERSON_FEMALE_ARCS, "gender": ("Female", "Male")},
        }
    )

    shapes = resolve_shapes(
        (
            "dcid:CountPerson",
            "dcid:DevelopmentFinance",
            "dcid:CountPersonFemale",
            "dcid:NotAKnownShape",
        ),
        graph=graph,
    )

    assert shapes.keys() == {"dcid:CountPerson"}


def test_resolve_shapes_repeats_the_same_shape_for_a_duplicate_ref():
    graph = FakeMintGraphReader(arcs={"CountPerson": _COUNT_PERSON_ARCS})

    shapes = resolve_shapes(("dcid:CountPerson", "dcid:CountPerson"), graph=graph)

    assert shapes == {"dcid:CountPerson": shapes["dcid:CountPerson"]}
    # A duplicate ref adds no graph read.
    assert graph.calls["node_arcs_many"] == 1


def test_resolve_shapes_of_an_empty_batch_makes_no_graph_call():
    graph = FakeMintGraphReader(arcs={})

    assert resolve_shapes((), graph=graph) == {}
    assert graph.calls["node_arcs_many"] == 0


class _UnresolvedArcGraphReader:
    """Marks unresolved arc targets as `UNRESOLVED_NODE`, as live `/v2/node` batches do."""

    def node_label(self, dcid: str) -> str | None:
        raise NotImplementedError

    def observation_facets_batch(
        self, *, stat_var: str, entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, list[Facet]]:
        raise NotImplementedError

    def declares_facet_defaults(self) -> bool:
        raise NotImplementedError

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        raise NotImplementedError

    def node_arcs(self, dcid: str, *, properties: Sequence[str]) -> dict[str, tuple[str, ...]]:
        raise NotImplementedError

    def node_subjects(self, dcid: str, *, property: str, filters=None, max_results: int):
        raise NotImplementedError

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        raise NotImplementedError

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        if "UnresolvedArcs" not in dcids:
            return {}
        return {
            "UnresolvedArcs": {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient", UNRESOLVED_NODE),
                "measuredProperty": ("amount",),
                "populationType": ("DevelopmentFinance",),
                "constraintProperties": ("DevelopmentFinanceScheme",),
                "DevelopmentFinanceScheme": (UNRESOLVED_NODE, "ODAGrants"),
            }
        }


def test_resolve_shapes_drops_the_unresolved_node_marker_like_node_arcs_drops_the_entry():
    shapes = resolve_shapes(("dcid:UnresolvedArcs",), graph=_UnresolvedArcGraphReader())

    shape = shapes["dcid:UnresolvedArcs"]
    assert shape.declared_roles == ("recipient",)
    assert shape.slots == (("DevelopmentFinanceScheme", "ODAGrants"),)
