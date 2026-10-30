"""Test identity projection, materialisation requests, and member resolution in `datasets_api.adapter`.
The member-resolution cases cover pinned identities; described-arm enumeration has separate tests.
"""

from __future__ import annotations

import pytest
from dfo_spec_kit.identity import IdentityError, build_projection

from code_index.shared import get_index
from datasets_api.adapter import (
    build_identity_document,
    provenance_pin,
    resolve_definition_members,
    to_materialisation_request,
)
from datasets_api.codes_materialise import (
    CurrencyRoleUndeclared,
    InvalidRoleRef,
    RoleIncompatible,
    SpecNotExportable,
)
from datasets_api.mint_models import FacetRefBinding, Scope, WindowRange
from datasets_api.shape import StatVarShape, resolve_shape
from materialise.recipes import RoleAxisConflictError
from spec_registry.identity import recompute_identity
from statvar_members import MemberEnumerator

from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds

_SHAPE = StatVarShape(
    stat_var="dcid:MalariaGrantDisbursements",
    declared_roles=("recipient",),
    slots=(("grantType", "Grant"), ("fundingStream", None)),
    measured_property="amount",
    population_type="DevelopmentFinance",
)


def test_identity_document_is_schema_valid_and_pins_the_bare_stat_var_id():
    document = build_identity_document(_SHAPE, scope=None)

    build_projection(document)  # raises IdentityError if invalid
    assert document["shape"] == {"kind": "stat_var", "id": "MalariaGrantDisbursements"}
    assert document["slots"] == []


def test_no_scope_canonicalises_to_unbound_means_everything():
    document = build_identity_document(_SHAPE, scope=None)

    assert document["entities"] == {}
    assert document["requested_window"] == {"kind": "all"}
    assert "facets" not in document


def test_scope_entities_and_facets_populate_the_document():
    scope = Scope(
        entities={"recipient": ["country/KEN"]},
        facets={
            "provenance": FacetRefBinding(ref="OECD_DAC2A"),
            "measurementMethod": FacetRefBinding(ref="GrantEquivalent"),
            "unit": FacetRefBinding(ref="USD"),
        },
        window=WindowRange(start_year=2015, end_year=2020),
    )

    document = build_identity_document(_SHAPE, scope=scope)

    assert document["entities"] == {"recipient": ["country/KEN"]}
    assert document["requested_window"] == {"kind": "range", "start_year": 2015, "end_year": 2020}
    assert document["facets"] == {
        "provenance": {"state": "value", "ref": "OECD_DAC2A"},
        "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
        "unit": {"state": "value", "ref": "USD"},
    }
    build_projection(document)


def test_the_minted_spec_id_is_insensitive_to_declared_roles_and_slots():
    # The pinned identity hashes only the bare StatVar ID, not its resolved roles or slots.
    narrow_shape = StatVarShape(
        stat_var="dcid:MalariaGrantDisbursements",
        declared_roles=("recipient",),
        slots=(("grantType", "Grant"),),
        measured_property="amount",
        population_type="DevelopmentFinance",
    )
    wide_shape = StatVarShape(
        stat_var="dcid:MalariaGrantDisbursements",
        declared_roles=("recipient", "provider"),
        slots=(("grantType", "Grant"), ("fundingStream", None)),
        measured_property="amount",
        population_type="DevelopmentFinance",
    )

    narrow_spec_id = recompute_identity(build_identity_document(narrow_shape, scope=None)).spec_id
    wide_spec_id = recompute_identity(build_identity_document(wide_shape, scope=None)).spec_id

    assert narrow_spec_id == wide_spec_id


def test_the_minted_spec_id_is_insensitive_to_the_graphs_own_arc_state():
    # Both graphs resolve the same bare dcid, so differing arcs leave the pinned identity unchanged.
    first_graph = FakeMintGraphReader(
        arcs={
            "MalariaGrantDisbursements": {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "constraintProperties": ("grantType",),
                "grantType": ("Grant",),
            }
        }
    )
    second_graph = FakeMintGraphReader(
        arcs={
            "MalariaGrantDisbursements": {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("provider", "recipient"),
                "constraintProperties": ("grantType", "fundingStream"),
                "grantType": ("Loan",),
                "fundingStream": ("Bilateral",),
            }
        }
    )

    first_document = build_identity_document(
        resolve_shape("dcid:MalariaGrantDisbursements", graph=first_graph), scope=None
    )
    second_document = build_identity_document(
        resolve_shape("dcid:MalariaGrantDisbursements", graph=second_graph), scope=None
    )

    assert recompute_identity(first_document).spec_id == recompute_identity(second_document).spec_id


def test_resolve_definition_members_refuses_spec_not_exportable_for_an_unanchored_described_arm():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "population_type": "Person"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    graph = FakeMintGraphReader()

    with pytest.raises(SpecNotExportable) as exc_info:
        resolve_definition_members(
            identity,
            spec_id="spec_test",
            graph=graph,
            enumerator=MemberEnumerator(),
            code_index=get_index(),
            bounds=mint_bounds(),
        )

    assert exc_info.value.spec_id == "spec_test"
    assert "measured_property" in str(exc_info.value)


def test_resolve_definition_members_refuses_loudly_on_a_dcid_prefixed_pin_id():
    # Shape IDs are bare dcids; a `dcid:` prefix fails schema validation before member resolution.
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "dcid:MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }
    graph = FakeMintGraphReader()

    with pytest.raises(IdentityError) as exc_info:
        resolve_definition_members(
            identity,
            spec_id="spec_test",
            graph=graph,
            enumerator=MemberEnumerator(),
            code_index=get_index(),
            bounds=mint_bounds(),
        )

    assert exc_info.value.category == "schema_violation"


def test_resolve_definition_members_round_trips_the_dcid_prefix_stripped_at_mint():
    graph = FakeMintGraphReader(
        arcs={
            "MalariaGrantDisbursements": {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            }
        }
    )
    minted_shape = resolve_shape("dcid:MalariaGrantDisbursements", graph=graph)
    document = build_identity_document(minted_shape, scope=None)
    assert document["shape"]["id"] == "MalariaGrantDisbursements"  # bare, no prefix

    resolved_shapes = resolve_definition_members(
        document,
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    ).shapes

    assert [shape.stat_var for shape in resolved_shapes] == ["dcid:MalariaGrantDisbursements"]


def test_scope_naming_an_undeclared_role_raises_invalid_role_ref():
    scope = Scope(entities={"donor": ["country/FRA"]})

    with pytest.raises(InvalidRoleRef):
        build_identity_document(_SHAPE, scope=scope)


def test_undeclared_role_naming_a_real_graph_property_raises_role_incompatible():
    # `provider` is a graph property, but `_SHAPE` does not declare it as a role.
    graph = FakeMintGraphReader(arcs={"provider": {"typeOf": ("Property",)}})
    scope = Scope(entities={"provider": ["org/USAID"]})

    with pytest.raises(RoleIncompatible) as exc_info:
        build_identity_document(_SHAPE, scope=scope, graph=graph)

    assert "provider" in str(exc_info.value)


def test_undeclared_role_naming_a_non_property_graph_node_raises_invalid_role_ref():
    # A `Class` node does not qualify as a role property, even when its dcid exists.
    graph = FakeMintGraphReader(arcs={"DevelopmentFinance": {"typeOf": ("Class",)}})
    scope = Scope(entities={"DevelopmentFinance": ["country/FRA"]})

    with pytest.raises(InvalidRoleRef):
        build_identity_document(_SHAPE, scope=scope, graph=graph)


def test_undeclared_role_naming_no_graph_node_still_raises_invalid_role_ref():
    graph = FakeMintGraphReader(arcs={})
    scope = Scope(entities={"donor": ["country/FRA"]})

    with pytest.raises(InvalidRoleRef):
        build_identity_document(_SHAPE, scope=scope, graph=graph)


def test_scope_naming_observation_about_on_a_role_declaring_statvar_raises_role_incompatible():
    scope = Scope(entities={"observationAbout": ["country/FRA"]})

    with pytest.raises(RoleIncompatible) as exc_info:
        build_identity_document(_SHAPE, scope=scope)

    assert "observationAbout" in str(exc_info.value)


# Roles and constraint slots are independent, so a role can be pinned by a slot.
# This shape also has an unpinned `provider` role.
_PINNED_ROLE_SHAPE = StatVarShape(
    stat_var="dcid:Health-ODAGrants-ETH",
    declared_roles=("recipient", "provider"),
    slots=(
        ("recipient", "country/ETH"),
        ("DevelopmentFinanceScheme", "ODAGrants"),
    ),
    measured_property="amount",
    population_type="DevelopmentFinance",
)


def test_scope_rebinding_a_slot_pinned_role_to_a_different_value_refuses():
    scope = Scope(entities={"recipient": ["country/KEN"]})

    with pytest.raises(RoleIncompatible) as exc_info:
        build_identity_document(_PINNED_ROLE_SHAPE, scope=scope)

    assert "recipient" in str(exc_info.value)
    assert "country/ETH" in str(exc_info.value)


def test_scope_rebinding_a_slot_pinned_role_to_the_same_value_still_refuses():
    scope = Scope(entities={"recipient": ["country/ETH"]})

    with pytest.raises(RoleIncompatible) as exc_info:
        build_identity_document(_PINNED_ROLE_SHAPE, scope=scope)

    assert "recipient" in str(exc_info.value)
    assert "country/ETH" in str(exc_info.value)


def test_scope_binding_a_declared_unpinned_role_still_succeeds():
    scope = Scope(entities={"provider": ["org/USAID"]})

    document = build_identity_document(_PINNED_ROLE_SHAPE, scope=scope)

    assert document["entities"] == {"provider": ["org/USAID"]}


def test_scope_rebinding_a_pinned_role_via_its_axis_alias_refuses():
    # `destination` aliases pinned `recipient`, so it must hit the same pin.
    # Pinning uses the declared property as its key.
    scope = Scope(entities={"destination": ["country/KEN"]})

    with pytest.raises(RoleIncompatible) as exc_info:
        build_identity_document(_PINNED_ROLE_SHAPE, scope=scope)

    assert "destination" in str(exc_info.value)
    assert "country/ETH" in str(exc_info.value)


def test_scope_rebinding_a_pinned_role_via_its_axis_alias_to_the_pin_value_still_refuses():
    # Pinned roles reject a redundant binding through an axis alias too.
    scope = Scope(entities={"destination": ["country/ETH"]})

    with pytest.raises(RoleIncompatible) as exc_info:
        build_identity_document(_PINNED_ROLE_SHAPE, scope=scope)

    assert "destination" in str(exc_info.value)
    assert "country/ETH" in str(exc_info.value)


def test_two_scope_keys_resolving_to_one_declared_role_refuse():
    # `provider` and its `origin` alias resolve to one role; accepting both would drop a binding.
    scope = Scope(entities={"origin": ["org/USAID"], "provider": ["org/BMZ"]})

    with pytest.raises(RoleIncompatible) as exc_info:
        build_identity_document(_PINNED_ROLE_SHAPE, scope=scope)

    message = str(exc_info.value)
    assert "origin" in message
    assert "provider" in message
    assert exc_info.value.role == "provider"


def test_two_scope_keys_sharing_one_axis_resolve_to_DIFFERENT_declared_roles_and_succeed():
    # `origin` and `destination` bind distinct roles on an unpinned two-role shape.
    shape = StatVarShape(
        stat_var="dcid:BilateralGrants",
        declared_roles=("recipient", "provider"),
        slots=(("DevelopmentFinanceScheme", "ODAGrants"),),
        measured_property="amount",
        population_type="DevelopmentFinance",
    )
    scope = Scope(
        entities={"origin": ["org/USAID"], "destination": ["org/ECHO"]},  # provider/recipient
    )

    document = build_identity_document(shape, scope=scope)

    assert document["entities"] == {"origin": ["org/USAID"], "destination": ["org/ECHO"]}


# This shape maps both `provider` and `creditor` to the `origin` axis.
_AMBIGUOUS_AXIS_SHAPE = StatVarShape(
    stat_var="dcid:BilateralLoans",
    declared_roles=("provider", "creditor"),
    slots=(),
    measured_property="amount",
    population_type="DevelopmentFinance",
)


def test_scope_binding_an_ambiguous_axis_refuses_with_an_arc_pointing_message():
    # Both declared roles map to `origin`, so the error must name the ambiguity and both roles.
    scope = Scope(entities={"origin": ["org/USAID"]})

    with pytest.raises(InvalidRoleRef) as exc_info:
        build_identity_document(_AMBIGUOUS_AXIS_SHAPE, scope=scope)

    message = str(exc_info.value)
    assert "two roles" in message
    # The message names both roles and the `observationProperties` source.
    assert "provider" in message
    assert "creditor" in message
    assert "observationProperties" in message
    assert exc_info.value.role == "origin"


def test_scope_binding_an_unresolvable_key_on_an_ambiguous_axis_shape_keeps_the_typo_message():
    # An unrelated key is a typo, not an ambiguous axis.
    scope = Scope(entities={"donor": ["org/USAID"]})

    with pytest.raises(InvalidRoleRef) as exc_info:
        build_identity_document(_AMBIGUOUS_AXIS_SHAPE, scope=scope)

    assert "does not declare" in str(exc_info.value)


@pytest.mark.parametrize(
    ("shape", "keys"),
    [
        (_SHAPE, ["destination", "recipient"]),
        # `origin` names both roles, so it binds neither and is not offered.
        (_AMBIGUOUS_AXIS_SHAPE, ["provider", "creditor"]),
    ],
)
def test_undeclared_role_refusal_offers_every_key_that_binds_a_declared_role(shape, keys):
    with pytest.raises(InvalidRoleRef) as exc_info:
        build_identity_document(shape, scope=Scope(entities={"donor": ["country/FRA"]}))

    (issue,) = exc_info.value.issues
    assert [a.ref for a in issue.alternatives] == keys
    for key in keys:
        build_identity_document(shape, scope=Scope(entities={key: ["country/FRA"]}))


def test_scope_binding_a_flow_axis_alias_of_a_declared_role_succeeds():
    # `destination` aliases `_SHAPE`'s declared `recipient` role.
    scope = Scope(entities={"destination": ["country/KEN"]})

    document = build_identity_document(_SHAPE, scope=scope)

    assert document["entities"] == {"destination": ["country/KEN"]}


def test_scope_binding_an_axis_alias_no_declared_role_places_still_refuses():
    # `_SHAPE` has no declared role on the `origin` axis.
    scope = Scope(entities={"origin": ["org/USAID"]})

    with pytest.raises(InvalidRoleRef):
        build_identity_document(_SHAPE, scope=scope)


def test_to_materialisation_request_round_trips_entities_facets_and_window():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {"recipient": ["country/KEN"]},
        "facets": {
            "provenance": {"state": "value", "ref": "OECD_DAC2A"},
            "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
            "unit": {"state": "value", "ref": "USD"},
        },
        "requested_window": {"kind": "range", "start_year": 2015, "end_year": 2020},
    }

    request = to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    assert request.stat_var == "dcid:MalariaGrantDisbursements"
    assert request.declared_roles == ("recipient",)
    assert request.role_bindings == {"recipient": ["country/KEN"]}
    assert request.facet_bindings["provenance"].state == "bound"
    assert request.facet_bindings["provenance"].ref == "OECD_DAC2A"
    assert request.window.kind == "range"
    assert request.window.start_year == 2015
    assert request.window.end_year == 2020


@pytest.mark.parametrize(
    ("refs", "bound"),
    [
        (["OECD_DAC2A", "OECD_DAC2A"], "OECD_DAC2A"),
        (["OECD_DAC2A", "OECD_CRS"], None),
    ],
    ids=["one-distinct-ref", "several-refs"],
)
def test_to_materialisation_request_binds_a_provenance_set_of_one_and_leaves_a_wider_set_unbound(
    refs: list[str], bound: str | None
):
    """A wider set is read under its refs' scope (`pinned_provenance_scope`) and elects
    among them, so it binds no ref for selection."""
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "facets": {"provenance": {"state": "set", "refs": refs}},
        "requested_window": {"kind": "all"},
    }

    request = to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    binding = request.facet_bindings.get("provenance")
    assert (None if binding is None else (binding.state, binding.ref)) == (
        None if bound is None else ("bound", bound)
    )
    assert provenance_pin(identity) == frozenset(refs)


def test_to_materialisation_request_translates_an_axis_alias_entity_role_to_the_declared_role():
    # Translate the `destination` alias to declared `recipient` for SDMX role filters.
    # Keep the stored key unchanged because it contributes to `spec_id`.
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {"destination": ["country/KEN"]},
        "requested_window": {"kind": "all"},
    }

    request = to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    assert request.role_bindings == {"recipient": ["country/KEN"]}


def test_to_materialisation_request_defaults_to_unbound_entities_and_facets():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "requested_window": {"kind": "all"},
    }

    request = to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    assert request.role_bindings == {}
    assert request.facet_bindings == {}
    assert request.window.kind == "all"


def test_to_materialisation_request_maps_an_iso_currency_request_to_a_request_binding():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "facets": {
            "unit": {
                "state": "value",
                "request": {"currency": "USD", "prices": "current"},
            }
        },
        "requested_window": {"kind": "all"},
    }

    request = to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    binding = request.facet_bindings["unit"]
    assert binding.state == "request"
    assert binding.ref is None
    assert binding.request is not None
    assert binding.request.currency == "USD"
    assert binding.request.prices == "current"
    assert binding.request.base_year is None


def test_to_materialisation_request_maps_a_constant_price_request_with_base_year():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "facets": {
            "unit": {
                "state": "value",
                "request": {"currency": "EUR", "prices": "constant", "base_year": 2020},
            }
        },
        "requested_window": {"kind": "all"},
    }

    request = to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    binding = request.facet_bindings["unit"]
    assert binding.request.currency == "EUR"
    assert binding.request.prices == "constant"
    assert binding.request.base_year == 2020


def test_to_materialisation_request_maps_a_role_shaped_currency_the_shape_declares():
    role_shape = StatVarShape(
        stat_var="dcid:BilateralFlow",
        # `financingProvider` and `financingRecipient` map to `origin` and `destination`.
        declared_roles=("financingProvider", "financingRecipient"),
        slots=(),
        measured_property="amount",
        population_type="DevelopmentFinance",
    )
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "BilateralFlow"},
        "slots": [],
        "entities": {},
        "facets": {
            "unit": {
                "state": "value",
                "request": {"currency": {"role": "origin"}, "prices": "current"},
            }
        },
        "requested_window": {"kind": "all"},
    }

    request = to_materialisation_request(identity, shape=role_shape, code_index=get_index())

    assert request.facet_bindings["unit"].request.currency == "origin"


def test_to_materialisation_request_refuses_a_role_shaped_currency_the_shape_lacks():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "facets": {
            "unit": {
                "state": "value",
                "request": {"currency": {"role": "origin"}, "prices": "current"},
            }
        },
        "requested_window": {"kind": "all"},
    }

    with pytest.raises(CurrencyRoleUndeclared) as exc_info:
        to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    # Direct adapter calls leave `pointer` unset; mint doors attach it later.
    assert exc_info.value.pointer is None
    assert exc_info.value.role == "origin"
    assert exc_info.value.declared_roles == ("recipient",)


def test_to_materialisation_request_refuses_a_role_shaped_currency_naming_an_ambiguous_axis():
    # A same-axis shape (`provider`/`creditor` both on `origin`) carries a
    # role on the axis the currency's `role` names -- twice -- so a message
    # claiming no role is declared there would be false.
    # `graph_role_for_spec_role` raises RoleAxisConflictError and the adapter
    # must translate it to CurrencyRoleUndeclared naming both roles.
    shape = StatVarShape(
        stat_var="dcid:BilateralLoans",
        declared_roles=("provider", "creditor"),
        slots=(),
        measured_property="amount",
        population_type="DevelopmentFinance",
    )
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "BilateralLoans"},
        "slots": [],
        "entities": {},
        "facets": {
            "unit": {
                "state": "value",
                "request": {"currency": {"role": "origin"}, "prices": "current"},
            }
        },
        "requested_window": {"kind": "all"},
    }

    with pytest.raises(CurrencyRoleUndeclared) as exc_info:
        to_materialisation_request(identity, shape=shape, code_index=get_index())

    message = str(exc_info.value)
    assert "provider" in message
    assert "creditor" in message
    assert "origin" in message
    assert exc_info.value.role == "origin"
    assert exc_info.value.declared_roles == ("provider", "creditor")
    # Neither role is reachable through a currency role, so none is offered.
    assert exc_info.value.alternatives == []
    assert isinstance(exc_info.value.__cause__, RoleAxisConflictError)


def test_to_materialisation_request_refuses_a_unit_facet_carrying_both_ref_and_request():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "facets": {
            "unit": {
                "state": "value",
                "ref": "USD",
                "request": {"currency": "USD", "prices": "current"},
            }
        },
        "requested_window": {"kind": "all"},
    }

    with pytest.raises(IdentityError):
        to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())


def test_to_materialisation_request_refuses_a_unit_facet_carrying_neither_ref_nor_request():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "facets": {"unit": {"state": "value"}},
        "requested_window": {"kind": "all"},
    }

    with pytest.raises(IdentityError):
        to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())


def test_to_materialisation_request_leaves_a_ref_form_unit_facet_unchanged():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "facets": {"unit": {"state": "value", "ref": "USD"}},
        "requested_window": {"kind": "all"},
    }

    request = to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    binding = request.facet_bindings["unit"]
    assert binding.state == "bound"
    assert binding.ref == "USD"
    assert binding.request is None


def test_to_materialisation_request_binds_a_null_ref_to_facets_without_the_axis():
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "stat_var", "id": "MalariaGrantDisbursements"},
        "slots": [],
        "entities": {},
        "facets": {"measurementMethod": {"state": "value", "ref": None}},
        "requested_window": {"kind": "all"},
    }

    request = to_materialisation_request(identity, shape=_SHAPE, code_index=get_index())

    binding = request.facet_bindings["measurementMethod"]
    assert (binding.state, binding.ref, binding.request) == ("bound", None, None)
