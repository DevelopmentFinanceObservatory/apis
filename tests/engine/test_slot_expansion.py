"""Test CRS binding expansion, declaration versions, and identity preservation."""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import Facet
from search.engine.retrieval.bound_values import (
    bound_values_from_bindings,
    declaration_version,
    slot_expansions,
)
from search.engine.retrieval.retrieve import BoundValuesUnrealisable, Materialised, materialise
from search.engine.shape import ShapeDraft
from search.models import BaseIdentity, Expansion
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

# Real CRS codes from the fixture index.

_HEALTH = "CRSPurposeCode120"  # hierarchical (crsPurpose), 23-node closure
_MALARIA_CONTROL = "CRSPurposeCode12262"  # a leaf under _HEALTH
_FLOW_GRANT = "CRSTypeOfFlow10"  # crsTypeOfFlow is flat (A4c)


def _binding(prop: str, values: list[str], *, kind: str = "value", evidence: str = "mentioned"):
    return SlotBindingDraft(
        axis="how", property_dcid=prop, kind=kind, value_dcids=values, evidence=evidence
    )


def _arcs(**constraints: str) -> dict:
    return {
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{prop: {"nodes": [{"dcid": val}]} for prop, val in constraints.items()},
    }


# Expansion mechanics.


def test_a_hierarchical_value_binding_expands_to_its_subtree_without_mutating_input():
    binding = _binding("crsPurpose", [_HEALTH])
    bound_values = bound_values_from_bindings([binding])
    added = set(bound_values["crsPurpose"])
    assert _HEALTH in added
    assert _MALARIA_CONTROL in added
    assert len(added) == len(get_index().descendants(_HEALTH))
    assert binding.value_dcids == [_HEALTH]


def test_a_hierarchical_set_binding_expands_every_member_value():
    bound_values = bound_values_from_bindings(
        [_binding("crsPurpose", [_HEALTH, _MALARIA_CONTROL], kind="set")]
    )
    # The leaf is already in the parent's closure, so expansion stays unique.
    assert set(bound_values["crsPurpose"]) == set(get_index().descendants(_HEALTH))


def test_expansion_over_a_flat_system_is_a_genuine_no_op():
    """A4c. crsTypeOfFlow is flat and unchanged."""
    bound_values = bound_values_from_bindings([_binding("crsTypeOfFlow", [_FLOW_GRANT])])
    assert bound_values["crsTypeOfFlow"] == [_FLOW_GRANT]


def test_expansion_over_a_property_the_index_does_not_carry_is_a_no_op():
    """The CRS-only index leaves a bound gender value unchanged."""
    bound_values = bound_values_from_bindings([_binding("gender", ["Female"])])
    assert bound_values["gender"] == ["Female"]


def test_expansion_adds_values_under_an_existing_key_and_adds_no_keys():
    """Expansion adds descendants under bound properties and introduces no keys."""
    before = {"crsPurpose", "crsTypeOfFlow", "gender"}
    bindings = [
        _binding("crsPurpose", [_HEALTH]),
        _binding("crsTypeOfFlow", [_FLOW_GRANT]),
        _binding("gender", ["Female"]),
    ]
    bound_values = bound_values_from_bindings(bindings)
    assert set(bound_values) == before
    assert set(bound_values["crsPurpose"]) >= {_HEALTH}


def test_slot_expansions_reports_only_what_was_added():
    expansions = slot_expansions(
        [
            _binding("crsPurpose", [_HEALTH]),
            _binding("crsTypeOfFlow", [_FLOW_GRANT]),
            _binding("gender", ["Female"]),
        ]
    )
    assert set(expansions) == {"crsPurpose"}
    health_expansion = expansions["crsPurpose"]
    assert isinstance(health_expansion, Expansion)
    added_dcids = {ref.ref for ref in health_expansion.added}
    assert _HEALTH not in added_dcids  # Expansion metadata lists descendants only.
    assert _MALARIA_CONTROL in added_dcids
    malaria_ref = next(ref for ref in health_expansion.added if ref.ref == _MALARIA_CONTROL)
    assert malaria_ref.label == "malaria control"


def test_slot_expansions_is_empty_when_nothing_was_added():
    expansions = slot_expansions(
        [_binding("crsTypeOfFlow", [_FLOW_GRANT]), _binding("gender", ["Female"])]
    )
    assert expansions == {}


# A described identity whose five-tuple the fixture index does not carry, so the graph
# decides its membership.
_UNINDEXED_IDENTITY = BaseIdentity.model_validate(
    {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "EconomicActivity",
            "measured_property": "amount",
            "stat_type": "measuredValue",
        },
        "slots": [],
        "requested_window": {"kind": "all"},
    }
)


def test_declaration_version_is_the_index_content_hash_when_a_declared_system_was_bound():
    assert (
        declaration_version([_binding("crsPurpose", [_HEALTH])], _UNINDEXED_IDENTITY)
        == get_index().stamp.content_hash
    )
    # The hash also identifies consulted flat systems.
    assert (
        declaration_version([_binding("crsTypeOfFlow", [_FLOW_GRANT])], _UNINDEXED_IDENTITY)
        == get_index().stamp.content_hash
    )


def test_declaration_version_is_none_when_the_index_decides_nothing():
    assert declaration_version([_binding("gender", ["Female"])], _UNINDEXED_IDENTITY) is None
    assert declaration_version([], _UNINDEXED_IDENTITY) is None


# Parent-code expansion through role-bearing resolution.

_SV_ROLE = "Amount_EconomicActivity_CrsFixture"
_DESTINATION = "country/KEN"
_ORIGIN = "country/FRA"


def _role_shape(*, crs_purpose: str) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_crs_role_shape",
        label="test CRS shape",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _SV_ROLE: {
                **_arcs(crsPurpose=crs_purpose),
                "observationProperties": {
                    "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
                },
            }
        },
        declared_roles=("financingProvider", "financingRecipient"),
        representative_score=0.9,
    )


def _role_sdmx() -> FakeSdmxClient:
    role_tuple = (
        ("financingProvider", _ORIGIN),
        ("financingRecipient", _DESTINATION),
    )
    return FakeSdmxClient(
        observations={
            _SV_ROLE: {
                role_tuple: [
                    Facet(
                        earliest_date="2015",
                        latest_date="2020",
                        obs_count=5,
                        dates=["2020"],
                        values=[1.0],
                        facet_id="f1",
                    )
                ]
            }
        }
    )


def test_a_parent_code_crs_ask_returns_data_instead_of_no_data():
    """Health at purpose 120: no member asserts the parent code directly (the
    fixture's only member is coded to the malaria-control leaf), so before
    expansion graph_confirm_resolve's own filter drops it and returns
    binding_mismatch (see test_parent_code_without_expansion_is_a_binding_mismatch
    below, which proves the pre-expansion map really does exclude this
    member). After expansion the leaf is in bound_values["crsPurpose"], the
    member survives the filter, and the role-keyed read confirms it."""
    shape = _role_shape(crs_purpose=_MALARIA_CONTROL)
    bindings = [_binding("crsPurpose", [_HEALTH])]

    result = materialise(
        shape,
        bindings,
        {"financingProvider": [_ORIGIN], "financingRecipient": [_DESTINATION]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=_role_sdmx(),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_ROLE]


def test_parent_code_without_expansion_would_be_a_binding_mismatch():
    """Without descendant expansion, the malaria-control leaf falls outside health's binding."""
    from search.engine.retrieval.bound_values import _collapse_bindings

    pre_expansion = _collapse_bindings([_binding("crsPurpose", [_HEALTH])])
    assert _MALARIA_CONTROL not in pre_expansion["crsPurpose"]
    post_expansion = bound_values_from_bindings([_binding("crsPurpose", [_HEALTH])])
    assert _MALARIA_CONTROL in post_expansion["crsPurpose"]


# Standard-path expansion uses a synthetic shape because CRS queries are role-bearing.


def _crs_purpose_shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_standard_crs_purpose_shape",
        label="test shape",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV_ROLE: _arcs(crsPurpose=_MALARIA_CONTROL)},
        representative_score=0.9,
    )


def _graph_with_crs_data() -> FakeGraph:
    obs = {
        f"{_SV_ROLE}|{_DESTINATION}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
        ]
    }
    return FakeGraph(nodes={}, obs=obs, detect={}, resolve={})


def test_standard_path_reads_a_member_in_the_subtree():
    shape = _crs_purpose_shape()
    bindings = [_binding("crsPurpose", [_HEALTH])]

    result = materialise(
        shape,
        bindings,
        {"observationAbout": [_DESTINATION]},
        graph=_graph_with_crs_data(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_SV_ROLE]


def test_standard_path_refuses_a_sibling_outside_the_subtree():
    """An education leaf stays outside health's subtree, so no member holds the value."""
    shape = ShapeDraft(
        shape_id="test_standard_crs_sibling_shape",
        label="test shape",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV_ROLE: _arcs(crsPurpose="CRSPurposeCode11110")},
        representative_score=0.9,
    )
    bindings = [_binding("crsPurpose", [_HEALTH])]

    result = materialise(
        shape,
        bindings,
        {"observationAbout": [_DESTINATION]},
        graph=_graph_with_crs_data(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert isinstance(result, BoundValuesUnrealisable)
    assert [c.properties for c in result.conflicts] == [{"crsPurpose"}]
