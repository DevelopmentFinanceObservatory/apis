"""Test spec1 identity mapping, canonicalisation, and pinned hashes."""

from __future__ import annotations

import hashlib
from typing import ClassVar

import pytest
import rfc8785
from dfo_spec_kit import identity as _identity_core
from dfo_spec_kit.identity import IdentityError

from search.engine.spec_id import (
    _slot_binding_member,
    bound_facet_axes,
    build_identity_projection,
    build_spec_identity,
    compute_spec_id,
    mint_identity,
)
from search.models import (
    Axis,
    BaseIdentity,
    BindingSet,
    BindingUnbound,
    BindingValue,
    Entity,
    EntityGroup,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    RoleAxis,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from tests.engine._base_identity import DECLARING_SHAPE, shape_for

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _slot_key(axis: Axis, prop_dcid: str | None, label: str = "") -> SlotKey:
    return SlotKey(
        axis=axis,
        property=GraphRef(dcid=prop_dcid, label=label) if prop_dcid else None,
        label=label or prop_dcid or axis,
    )


def _value_slot(axis: Axis, prop_dcid: str | None, value_dcid: str) -> Slot:
    return Slot(
        key=_slot_key(axis, prop_dcid),
        binding=BindingValue(
            value=SlotValue(
                ref=GraphRef(dcid=value_dcid, label=value_dcid),
                value_kind="enum_value",
            )
        ),
    )


def _set_slot(axis: Axis, prop_dcid: str | None, value_dcids: list[str]) -> Slot:
    return Slot(
        key=_slot_key(axis, prop_dcid),
        binding=BindingSet(
            values=[
                SlotValue(ref=GraphRef(dcid=d, label=d), value_kind="enum_value")
                for d in value_dcids
            ]
        ),
    )


def _unbound_slot(axis: Axis, prop_dcid: str | None) -> Slot:
    return Slot(key=_slot_key(axis, prop_dcid), binding=BindingUnbound())


def _absent_slot(axis: Axis, prop_dcid: str | None) -> Slot:
    """Build the unbound wire form for an inapplicable slot.

    spec1 has no separate absent state.
    """
    return Slot(key=_slot_key(axis, prop_dcid), binding=BindingUnbound())


def _when_slot_value(start: int | None, end: int | None) -> Slot:
    """Build a time slot; requested_window alone contributes time to the hash."""
    tw = TimeWindow(start_year=start, end_year=end)
    return Slot(
        key=_slot_key("when", None, "time window"),
        binding=BindingValue(value=SlotValue(ref=None, value_kind="time_window", time_window=tw)),
    )


def _subject_entity(dcid: str) -> Entity:
    return Entity(ref=GraphRef(dcid=dcid, label=dcid), entity_type=None, role=EntityRoleSubject())


def _directional_entity(dcid: str, axis: RoleAxis, role_dcid: str = "irrelevant") -> Entity:
    """The role property is provenance; the axis determines identity."""
    return Entity(
        ref=GraphRef(dcid=dcid, label=dcid),
        entity_type=None,
        role=EntityRoleDirectional(role=GraphRef(dcid=role_dcid, label=role_dcid), axis=axis),
    )


# ---------------------------------------------------------------------------
# spec_ ids use a 32-character SHA-256 prefix.
# ---------------------------------------------------------------------------

_PIN_SLOTS = [
    _value_slot("what", "DevelopmentFinanceScheme", "ODAGrants"),
    _value_slot("how", "DevelopmentFinancePurpose", "DAC/Health"),
]
_PIN_ENTITIES = [
    _directional_entity("country/USA", "origin", "observationAbout"),
    _directional_entity("country/ETH", "destination", "DevelopmentFinanceRecipient"),
]


def test_pin_known_input_exact_hash() -> None:
    """The hash changes only when spec1's normalization, JCS, or SHA-256 rules rotate."""
    result = compute_spec_id(DECLARING_SHAPE, _PIN_SLOTS, entities=_PIN_ENTITIES, open_roles=())
    assert result == "spec_ecd79c14914a8d66bbf557969cdf3fa2"


# ---------------------------------------------------------------------------
# Slot order and set-member order independence
# ---------------------------------------------------------------------------


def test_slot_order_independence() -> None:
    """Slots in different orders must produce the same spec_id."""
    a = _value_slot("what", "PropA", "ValA")
    b = _value_slot("how", "PropB", "ValB")
    c = _value_slot("how", "PropC", "ValC")
    id_abc = compute_spec_id(shape_for("shape"), [a, b, c], open_roles=())
    id_cba = compute_spec_id(shape_for("shape"), [c, b, a], open_roles=())
    id_bac = compute_spec_id(shape_for("shape"), [b, a, c], open_roles=())
    assert id_abc == id_cba == id_bac


def test_served_document_rfc8785_hash_reproduces_spec_id() -> None:
    """#572. The kit's internal sort masks unsorted served slots.

    Hash the served RFC 8785 document directly; input order is Z before A.
    """
    z_first = _value_slot("what", "ZLastProp", "ValZ")
    a_second = _value_slot("how", "AFirstProp", "ValA")
    spec_id, base_identity = build_spec_identity(
        shape_for("shape"), [z_first, a_second], open_roles=()
    )

    assert [s.property for s in base_identity.slots] == ["AFirstProp", "ZLastProp"]

    canonical = rfc8785.dumps(base_identity.model_dump(exclude_none=True))
    digest = hashlib.sha256(canonical).hexdigest()
    assert spec_id == f"spec_{digest[:32]}"


def _assert_served_document_reproduces_spec_id(spec_id: str, base_identity) -> None:
    """Hash the served BaseIdentity directly to verify its spec id."""
    canonical = rfc8785.dumps(base_identity.model_dump(exclude_none=True))
    digest = hashlib.sha256(canonical).hexdigest()
    assert spec_id == f"spec_{digest[:32]}"


def test_served_multi_dcid_entity_bucket_reproduces_spec_id() -> None:
    """#572. A multi-DCID role bucket reproduces its id from the served document."""
    slots = [_value_slot("what", "DevelopmentFinanceScheme", "ODAGrants")]
    entities = [
        _directional_entity("country/USA", "destination", "DevelopmentFinanceRecipient"),
        _directional_entity("country/ETH", "destination", "DevelopmentFinanceRecipient"),
        _directional_entity("country/KEN", "destination", "DevelopmentFinanceRecipient"),
    ]
    spec_id, base_identity = build_spec_identity(
        shape_for("directional_roles"), slots, entities=entities, open_roles=()
    )
    assert base_identity.entities["destination"] == ["country/ETH", "country/KEN", "country/USA"]
    _assert_served_document_reproduces_spec_id(spec_id, base_identity)


def test_served_multi_entity_request_reproduces_spec_id() -> None:
    """#572. An origin-and-destination request reproduces its id from the served document."""
    entities = [
        _directional_entity("country/USA", "destination", "DevelopmentFinanceRecipient"),
        _directional_entity("country/KEN", "origin", "observationAbout"),
        _directional_entity("country/ETH", "destination", "DevelopmentFinanceRecipient"),
    ]
    spec_id, base_identity = build_spec_identity(
        shape_for("directional_roles"), [], entities=entities, open_roles=()
    )
    assert base_identity.entities["destination"] == ["country/ETH", "country/USA"]
    assert base_identity.entities["origin"] == ["country/KEN"]
    _assert_served_document_reproduces_spec_id(spec_id, base_identity)


@pytest.mark.parametrize("open_roles", [["origin"], ["origin", "origin"], []])
def test_open_roles_round_trip_to_the_identity_cores_spec_id(open_roles: list[RoleAxis]) -> None:
    """An identity with open roles hashes to the id the identity core gives its document, and
    an empty list mints the id of an identity without the member."""
    entities = [_directional_entity("country/NGA", "destination", "debtor")]
    shape = shape_for("directional_roles")
    spec_id, base_identity = build_spec_identity(
        shape, [], entities=entities, open_roles=open_roles
    )
    document = base_identity.model_dump(mode="json", exclude_none=True)

    assert spec_id == _identity_core.spec_id_from_value(_identity_core.build_projection(document))
    _assert_served_document_reproduces_spec_id(spec_id, base_identity)
    if open_roles:
        assert document["open_roles"] == ["origin"]
        assert spec_id != build_spec_identity(shape, [], entities=entities, open_roles=())[0]
    else:
        assert "open_roles" not in document


@pytest.mark.parametrize(
    ("entities", "shape", "message"),
    [
        ({"origin": ["country/CHN"]}, None, "entities also binds"),
        ({}, {"kind": "stat_var", "id": "Debt_ByCreditor"}, "pinned shape"),
    ],
)
def test_base_identity_refuses_an_open_role_bound_or_pinned(entities, shape, message) -> None:
    _, base_identity = build_spec_identity(
        shape_for("directional_roles"), [], entities=None, open_roles=()
    )
    document = base_identity.model_dump(mode="json", exclude_none=True)
    document.update(entities=entities, open_roles=["origin"])
    if shape is not None:
        document.update(shape=shape, slots=[])

    with pytest.raises(ValueError, match=message):
        BaseIdentity.model_validate(document)


def test_entity_group_label_is_bounded() -> None:
    EntityGroup(label="x" * 120)
    with pytest.raises(ValueError, match="label"):
        EntityGroup(label="x" * 121)


def test_served_multi_value_slot_reproduces_spec_id() -> None:
    """#572. A multi-value slot reproduces its id from the served document."""
    slots = [
        _value_slot("what", "AFirstProp", "ValA"),
        _set_slot(
            "how",
            "DevelopmentFinancePurpose",
            ["DAC/Medicaleducationtraining", "DAC/Healtheducation"],
        ),
    ]
    spec_id, base_identity = build_spec_identity(
        shape_for("directional_roles"), slots, open_roles=()
    )
    set_slot = next(s for s in base_identity.slots if s.property == "DevelopmentFinancePurpose")
    assert isinstance(set_slot.binding, IdentityBindingSet)
    assert set_slot.binding.refs == ["DAC/Healtheducation", "DAC/Medicaleducationtraining"]
    _assert_served_document_reproduces_spec_id(spec_id, base_identity)


def test_served_facet_set_reproduces_spec_id() -> None:
    """#572. An out-of-order measurementMethod set reproduces its served id."""
    common = [_value_slot("what", "SomeProp", "SomeValue")]
    spec_id, base_identity = build_spec_identity(
        shape_for("shape"),
        common,
        explicit_facets={
            "measurementMethod": IdentityBindingSet(refs=["OECD_Estimate", "IMF_Estimate"])
        },
        open_roles=(),
    )
    assert base_identity.facets is not None
    facet = base_identity.facets["measurementMethod"]
    assert isinstance(facet, IdentityBindingSet)
    assert facet.refs == ["IMF_Estimate", "OECD_Estimate"]
    _assert_served_document_reproduces_spec_id(spec_id, base_identity)


def test_a_null_method_ref_mints_its_own_id_matching_the_spec_kit_vector() -> None:
    """A measurementMethod bound to a null ref pins the facets that declare no method. It is a
    pin, not an unbound axis: the id differs from the unpinned identity's, the served document
    carries the null ref, and the id equals the dfo-spec-kit vector
    `identity/facet-null-ref-measurement-method`."""
    shape = {"kind": "statvar_schema", "measured_property": "developmentFinanceFlow"}
    slots = [_value_slot("how", "DevelopmentFinancePurpose", "Malaria")]
    entities = [_directional_entity("country/FRA", "origin")]

    spec_id, base_identity = build_spec_identity(
        shape,
        slots,
        entities=entities,
        explicit_facets={"measurementMethod": IdentityFacetValueAbsent(ref=None)},
        open_roles=(),
    )
    unpinned_id, _ = build_spec_identity(shape, slots, entities=entities, open_roles=())

    assert spec_id == "spec_8972b8b9a034b8fa6754042c2fc02317"
    assert spec_id != unpinned_id
    assert base_identity.facets == {"measurementMethod": IdentityFacetValueAbsent(ref=None)}
    assert bound_facet_axes(slots, {"measurementMethod": IdentityFacetValueAbsent(ref=None)}) == {
        "measurementMethod"
    }
    _assert_served_document_reproduces_spec_id(spec_id, base_identity)


def test_served_duplicate_set_members_reproduce_spec_id() -> None:
    """#572. Served set bindings deduplicate duplicate DCIDs before hashing."""
    common = [_value_slot("what", "SomeProp", "SomeValue")]
    slots = [*common, _set_slot("source", None, ["OECD_DAC1", "OECD_DAC2A", "OECD_DAC1"])]
    spec_id, base_identity = build_spec_identity(shape_for("shape"), slots, open_roles=())
    assert base_identity.facets is not None
    provenance = base_identity.facets["provenance"]
    assert isinstance(provenance, IdentityBindingSet)
    assert provenance.refs == ["OECD_DAC1", "OECD_DAC2A"]
    _assert_served_document_reproduces_spec_id(spec_id, base_identity)


def test_served_singleton_set_collapses_and_reproduces_spec_id() -> None:
    """#572. A singleton set is served as a value before hashing."""
    common = [_value_slot("what", "SomeProp", "SomeValue")]
    slots = [*common, _set_slot("source", None, ["OECD_DAC1", "OECD_DAC1"])]
    spec_id, base_identity = build_spec_identity(shape_for("shape"), slots, open_roles=())
    assert base_identity.facets is not None
    provenance = base_identity.facets["provenance"]
    assert isinstance(provenance, IdentityBindingValue)
    assert provenance.ref == "OECD_DAC1"
    _assert_served_document_reproduces_spec_id(spec_id, base_identity)


def test_set_binding_order_independent() -> None:
    """Two set-bound slots with the same dcids in different order must hash
    identically."""
    set_ab = _set_slot(
        "how", "DevelopmentFinancePurpose", ["DAC/Healtheducation", "DAC/Medicaleducationtraining"]
    )
    set_ba = _set_slot(
        "how", "DevelopmentFinancePurpose", ["DAC/Medicaleducationtraining", "DAC/Healtheducation"]
    )
    id_ab = compute_spec_id(shape_for("directional_roles"), [set_ab], open_roles=())
    id_ba = compute_spec_id(shape_for("directional_roles"), [set_ba], open_roles=())
    assert id_ab == id_ba


def test_different_shape_ids_produce_different_spec_ids() -> None:
    slots = [_value_slot("what", "SomeProp", "SomeValue")]
    id_a = compute_spec_id(shape_for("shape_a"), slots, open_roles=())
    id_b = compute_spec_id(shape_for("shape_b"), slots, open_roles=())
    assert id_a != id_b


# Direct identity minting from a projection.


def test_mint_identity_matches_build_spec_identity_for_the_equivalent_slots() -> None:
    """Minting the same projection directly reproduces the spec id and definition."""
    slots = [
        _value_slot("what", "AFirstProp", "ValA"),
        _set_slot(
            "how",
            "DevelopmentFinancePurpose",
            ["DAC/Healtheducation", "DAC/Medicaleducationtraining"],
        ),
    ]
    expected_spec_id, expected_definition = build_spec_identity(
        shape_for("shape"), slots, open_roles=()
    )

    projection = build_identity_projection(shape_for("shape"), slots, open_roles=())
    spec_id, definition = mint_identity(projection)

    assert spec_id == expected_spec_id
    assert definition == expected_definition


# Inapplicable slots appear as unbound in identity.


def test_unbound_and_absent_converge() -> None:
    """An absent helper and an unbound helper produce the same wire state."""
    base = [_value_slot("how", "DevelopmentFinancePurpose", "DAC/Health")]
    unbound_id = compute_spec_id(
        shape_for("directional_roles"),
        [*base, _unbound_slot("what", "DevelopmentFinanceScheme")],
        open_roles=(),
    )
    absent_id = compute_spec_id(
        shape_for("directional_roles"),
        [*base, _absent_slot("what", "DevelopmentFinanceScheme")],
        open_roles=(),
    )
    assert unbound_id == absent_id


def test_unbound_sentinel_stable_across_other_changes() -> None:
    """Changing another slot leaves the unbound slot's hash contribution unchanged."""
    id_a = compute_spec_id(
        shape_for("directional_roles"),
        [
            _value_slot("how", "DevelopmentFinancePurpose", "DAC/Health"),
            _unbound_slot("what", "DevelopmentFinanceScheme"),
        ],
        open_roles=(),
    )
    id_b = compute_spec_id(
        shape_for("directional_roles"),
        [
            _value_slot("how", "DevelopmentFinancePurpose", "DAC/Malariacontrol"),
            _unbound_slot("what", "DevelopmentFinanceScheme"),
        ],
        open_roles=(),
    )
    assert id_a != id_b


# Entities carry recipient identity. Where slots carry binding details.


class TestWhereSlotNeverAffectsIdentity:
    def test_property_backed_where_slot_binding_ignored_when_entities_equal(self) -> None:
        """Two different recipient dcids bound on the where-slot, but the
        SAME entities list, must hash identically -- the where-slot's own
        value carries no identity weight at all."""
        common = [_value_slot("what", "DevelopmentFinanceScheme", "ODAGrants")]
        entities = [
            _directional_entity("country/ETH", "destination", "DevelopmentFinanceRecipient")
        ]
        id_a = compute_spec_id(
            shape_for("directional_roles"),
            [*common, _value_slot("where", "DevelopmentFinanceRecipient", "country/ETH")],
            entities=entities,
            open_roles=(),
        )
        id_b = compute_spec_id(
            shape_for("directional_roles"),
            [*common, _value_slot("where", "DevelopmentFinanceRecipient", "country/KEN")],
            entities=entities,
            open_roles=(),
        )
        assert id_a == id_b

    def test_property_less_where_slot_binding_ignored(self) -> None:
        """The self-describing subject where-slot (property=None) is equally
        excluded -- present or absent from the slots list, bound to any
        dcid, the id is unaffected when entities stay the same."""
        entities = [_subject_entity("country/IND")]
        slots_no_where = [_value_slot("how", "activitySource", "GrossDomesticProduction")]
        slots_with_where = [
            *slots_no_where,
            Slot(
                key=SlotKey(axis="where", property=None, label="place"),
                binding=BindingValue(
                    value=SlotValue(
                        ref=GraphRef(dcid="country/IND", label="India"), value_kind="entity"
                    )
                ),
            ),
        ]
        id_a = compute_spec_id(shape_for("shape"), slots_no_where, entities=entities, open_roles=())
        id_b = compute_spec_id(
            shape_for("shape"), slots_with_where, entities=entities, open_roles=()
        )
        assert id_a == id_b

    def test_different_recipients_are_distinguished_via_entities_only(self) -> None:
        """Recipient values distinguish ids through the entities field."""
        slots = [
            _value_slot("what", "DevelopmentFinanceScheme", "ODAGrants"),
            _value_slot("how", "DevelopmentFinancePurpose", "DAC/Health"),
        ]
        id_eth = compute_spec_id(
            shape_for("directional_roles"),
            slots,
            entities=[
                _directional_entity("country/ETH", "destination", "DevelopmentFinanceRecipient")
            ],
            open_roles=(),
        )
        id_ken = compute_spec_id(
            shape_for("directional_roles"),
            slots,
            entities=[
                _directional_entity("country/KEN", "destination", "DevelopmentFinanceRecipient")
            ],
            open_roles=(),
        )
        assert id_eth != id_ken


# requested_window carries time. When slots are excluded from identity.


class TestWhenSlotNeverAffectsIdentity:
    def test_when_slot_binding_ignored_regardless_of_value(self) -> None:
        slots_a = [_when_slot_value(2010, 2020)]
        slots_b = [_when_slot_value(2015, 2018)]
        assert compute_spec_id(shape_for("shape"), slots_a, open_roles=()) == compute_spec_id(
            shape_for("shape"), slots_b, open_roles=()
        )

    def test_when_slot_present_vs_absent_from_the_list_agrees(self) -> None:
        other = _value_slot("what", "SomeProp", "SomeValue")
        id_without = compute_spec_id(shape_for("shape"), [other], open_roles=())
        id_with = compute_spec_id(
            shape_for("shape"), [other, _when_slot_value(2015, 2018)], open_roles=()
        )
        assert id_without == id_with


# Source slots map to provenance facets.


class TestSourceSlotMapsToProvenanceFacet:
    @pytest.mark.parametrize(
        ("source_a", "source_b"),
        [
            pytest.param(
                _unbound_slot("source", None),
                _value_slot("source", None, "OECD_DAC2A"),
                id="bound-vs-unbound",
            ),
            pytest.param(
                _value_slot("source", None, "OECD_DAC2A"),
                _value_slot("source", None, "OECD_DAC1"),
                id="different-bound-sources",
            ),
        ],
    )
    def test_bound_source_forms_produce_distinct_ids(self, source_a: Slot, source_b: Slot) -> None:
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        assert compute_spec_id(
            shape_for("shape"), [*common, source_a], open_roles=()
        ) != compute_spec_id(shape_for("shape"), [*common, source_b], open_roles=())

    @pytest.mark.parametrize(
        ("slots_a", "slots_b"),
        [
            pytest.param(
                [_unbound_slot("source", None)],
                [_absent_slot("source", None)],
                id="explicit-absent-vs-unbound",
            ),
            pytest.param([], [_unbound_slot("source", None)], id="omitted-vs-unbound"),
        ],
    )
    def test_omitted_source_forms_converge(self, slots_a: list[Slot], slots_b: list[Slot]) -> None:
        """An inapplicable, omitted, or unbound source slot hashes identically."""
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        assert compute_spec_id(
            shape_for("shape"), [*common, *slots_a], open_roles=()
        ) == compute_spec_id(shape_for("shape"), [*common, *slots_b], open_roles=())

    def test_source_set_binding_order_independent(self) -> None:
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        id_ab = compute_spec_id(
            shape_for("shape"),
            [*common, _set_slot("source", None, ["OECD_DAC1", "OECD_DAC2A"])],
            open_roles=(),
        )
        id_ba = compute_spec_id(
            shape_for("shape"),
            [*common, _set_slot("source", None, ["OECD_DAC2A", "OECD_DAC1"])],
            open_roles=(),
        )
        assert id_ab == id_ba

    def test_duplicate_source_axis_slots_raise(self) -> None:
        """Duplicate source slots raise instead of losing a binding."""
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        slots = [
            *common,
            _value_slot("source", None, "OECD_DAC1"),
            _value_slot("source", None, "OECD_DAC2A"),
        ]
        with pytest.raises(ValueError, match="at most one source-axis slot"):
            compute_spec_id(shape_for("shape"), slots, open_roles=())


# Explicit measurementMethod and unit facets.


class TestExplicitFacetBindings:
    def test_measurement_method_ref_binding_changes_id(self) -> None:
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        id_unbound = compute_spec_id(shape_for("shape"), common, open_roles=())
        id_bound = compute_spec_id(
            shape_for("shape"),
            common,
            explicit_facets={"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")},
            open_roles=(),
        )
        assert id_unbound != id_bound

    def test_unit_ref_binding_changes_id(self) -> None:
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        id_unbound = compute_spec_id(shape_for("shape"), common, open_roles=())
        id_bound = compute_spec_id(
            shape_for("shape"),
            common,
            explicit_facets={"unit": IdentityBindingValue(ref="USD")},
            open_roles=(),
        )
        assert id_unbound != id_bound

    def test_unit_request_form_binding_changes_id(self) -> None:
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        id_unbound = compute_spec_id(shape_for("shape"), common, open_roles=())
        id_bound = compute_spec_id(
            shape_for("shape"),
            common,
            explicit_facets={
                "unit": IdentityFacetValueRequest(
                    request=IdentityUnitRequest(currency="USD", prices="current")
                )
            },
            open_roles=(),
        )
        assert id_unbound != id_bound

    def test_provenance_key_in_explicit_facets_is_ignored(self) -> None:
        """The source slot alone maps provenance into identity.

        Ignore explicit provenance to avoid hashing the same source twice.
        """
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        id_no_explicit = compute_spec_id(shape_for("shape"), common, open_roles=())
        id_with_ignored_provenance = compute_spec_id(
            shape_for("shape"),
            common,
            explicit_facets={"provenance": IdentityBindingValue(ref="OECD_DAC1")},
            open_roles=(),
        )
        assert id_no_explicit == id_with_ignored_provenance

    def test_measurement_method_and_source_slot_provenance_both_apply(self) -> None:
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        slots_with_source = [*common, _value_slot("source", None, "OECD_DAC1")]
        id_source_only = compute_spec_id(shape_for("shape"), slots_with_source, open_roles=())
        id_both = compute_spec_id(
            shape_for("shape"),
            slots_with_source,
            explicit_facets={"measurementMethod": IdentityBindingValue(ref="GrossDisbursement")},
            open_roles=(),
        )
        assert id_source_only != id_both

    def test_bound_facet_axes_reports_explicit_axes(self) -> None:
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        slots_with_source = [*common, _value_slot("source", None, "OECD_DAC1")]
        axes = bound_facet_axes(
            slots_with_source,
            {
                "measurementMethod": IdentityBindingValue(ref="GrossDisbursement"),
                "unit": IdentityBindingValue(ref="USD"),
            },
        )
        assert axes == frozenset({"provenance", "measurementMethod", "unit"})

    def test_bound_facet_axes_defaults_to_slot_only_derivation(self) -> None:
        """Omitting explicit facets derives bound axes from slots alone."""
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        slots_with_source = [*common, _value_slot("source", None, "OECD_DAC1")]
        assert bound_facet_axes(slots_with_source) == frozenset({"provenance"})

    def test_measurement_method_and_unit_round_trip_through_identity_core(self) -> None:
        """The identity core rehashes measurementMethod and unit refs to the same id.

        Both bindings survive model validation unchanged.
        """
        common = [_value_slot("what", "SomeProp", "SomeValue")]
        spec_id, base_identity = build_spec_identity(
            shape_for("shape"),
            common,
            explicit_facets={
                "measurementMethod": IdentityBindingValue(ref="GrossDisbursement"),
                "unit": IdentityBindingValue(ref="USD"),
            },
            open_roles=(),
        )
        assert base_identity.facets is not None
        assert base_identity.facets["measurementMethod"] == IdentityBindingValue(
            ref="GrossDisbursement"
        )
        assert base_identity.facets["unit"] == IdentityBindingValue(ref="USD")
        rehashed = _identity_core.spec_id_from_value(base_identity.model_dump(exclude_none=True))
        assert rehashed == spec_id
        # Repeated minting reproduces the same id.
        spec_id_again, _ = build_spec_identity(
            shape_for("shape"),
            common,
            explicit_facets={
                "measurementMethod": IdentityBindingValue(ref="GrossDisbursement"),
                "unit": IdentityBindingValue(ref="USD"),
            },
            open_roles=(),
        )
        assert spec_id_again == spec_id

    def test_unbound_slot_is_identity_bearing(self) -> None:
        """A declared unbound slot contributes to identity and survives rehashing.

        Omitting it produces a different spec id.
        """
        slots = [
            _value_slot("what", "BoundProp", "BoundValue"),
            _unbound_slot("how", "UnboundProp"),
        ]
        spec_id, base_identity = build_spec_identity(shape_for("shape"), slots, open_roles=())
        assert any(s.binding.state == "unbound" for s in base_identity.slots)
        rehashed = _identity_core.spec_id_from_value(base_identity.model_dump(exclude_none=True))
        assert rehashed == spec_id

        spec_id_without_unbound, _ = build_spec_identity(
            shape_for("shape"),
            [_value_slot("what", "BoundProp", "BoundValue")],
            open_roles=(),
        )
        assert spec_id_without_unbound != spec_id


# Malformed inputs raise before minting.


class TestFailLoudOnMalformedInput:
    def test_binding_value_with_no_ref_raises(self) -> None:
        """A BindingValue without a ref raises ValueError."""
        literal_slot = Slot(
            key=_slot_key("what", "SomeProp"),
            binding=BindingValue(value=SlotValue(ref=None, value_kind="literal", literal="foo")),
        )
        with pytest.raises(ValueError, match="no ref"):
            compute_spec_id(shape_for("shape"), [literal_slot], open_roles=())

    def test_binding_set_member_with_no_ref_raises(self) -> None:
        set_slot = Slot(
            key=_slot_key("how", "SomeProp"),
            binding=BindingSet(
                values=[
                    SlotValue(ref=GraphRef(dcid="a", label="a"), value_kind="enum_value"),
                    SlotValue(ref=None, value_kind="literal", literal="b"),
                ]
            ),
        )
        with pytest.raises(ValueError, match="no ref"):
            compute_spec_id(shape_for("shape"), [set_slot], open_roles=())

    def test_what_slot_with_no_property_raises(self) -> None:
        no_prop_slot = Slot(key=_slot_key("what", None, "label"), binding=BindingUnbound())
        with pytest.raises(ValueError, match="no constraint property"):
            compute_spec_id(shape_for("shape"), [no_prop_slot], open_roles=())

    def test_unknown_binding_type_raises_value_error(self) -> None:
        """An unknown binding type raises ValueError from the adapter helper."""
        with pytest.raises(ValueError, match="Unknown binding type"):
            _slot_binding_member(object())


# ---------------------------------------------------------------------------
# Entities keyed by role fold into the id
# ---------------------------------------------------------------------------


def test_entities_absent_none_and_empty_list_all_agree() -> None:
    """entities=None, entities=[], and omitting the kwarg entirely must all
    map to the same `{}` entities member."""
    slots = [_value_slot("what", "SomeProp", "SomeValue")]
    assert compute_spec_id(shape_for("shape"), slots, open_roles=()) == compute_spec_id(
        shape_for("shape"), slots, entities=None, open_roles=()
    )
    assert compute_spec_id(shape_for("shape"), slots, open_roles=()) == compute_spec_id(
        shape_for("shape"), slots, entities=[], open_roles=()
    )


def test_entities_same_role_bucket_order_independent() -> None:
    """Two entities in the SAME role bucket, in different input order, must
    hash identically."""
    slots = [_value_slot("what", "DevelopmentFinanceScheme", "ODAGrants")]
    e1 = _directional_entity("country/ETH", "destination", "DevelopmentFinanceRecipient")
    e2 = _directional_entity("country/KEN", "destination", "DevelopmentFinanceRecipient")
    id_ab = compute_spec_id(shape_for("directional_roles"), slots, entities=[e1, e2], open_roles=())
    id_ba = compute_spec_id(shape_for("directional_roles"), slots, entities=[e2, e1], open_roles=())
    assert id_ab == id_ba


@pytest.mark.parametrize(
    ("shape_name", "slots", "entities_a", "entities_b"),
    [
        pytest.param(
            "economicactivity_amount_measuredvalue_nominal",
            [_value_slot("how", "activitySource", "GrossDomesticProduction")],
            [_subject_entity("country/IND")],
            [_subject_entity("country/BRA")],
            id="different-subject-entities",
        ),
        pytest.param(
            "directional_roles",
            [],
            [_directional_entity("country/USA", "origin")],
            [_directional_entity("country/USA", "destination")],
            id="origin-vs-destination",
        ),
        pytest.param(
            "shape",
            [],
            [_subject_entity("country/USA")],
            [_directional_entity("country/USA", "origin")],
            id="subject-vs-origin",
        ),
    ],
)
def test_entity_role_buckets_remain_distinct(
    shape_name: str,
    slots: list[Slot],
    entities_a: list[Entity],
    entities_b: list[Entity],
) -> None:
    """Entity identity distinguishes entity values and their role buckets."""
    assert compute_spec_id(
        shape_for(shape_name), slots, entities=entities_a, open_roles=()
    ) != compute_spec_id(shape_for(shape_name), slots, entities=entities_b, open_roles=())


def test_directional_role_dcid_does_not_affect_identity() -> None:
    """Directional role GraphRefs are provenance; changing one leaves the id unchanged."""
    slots: list[Slot] = []
    id_a = compute_spec_id(
        shape_for("directional_roles"),
        slots,
        entities=[
            _directional_entity(
                "country/ETH", "destination", role_dcid="DevelopmentFinanceRecipient"
            ),
        ],
        open_roles=(),
    )
    id_b = compute_spec_id(
        shape_for("directional_roles"),
        slots,
        entities=[_directional_entity("country/ETH", "destination", role_dcid="someOtherProperty")],
        open_roles=(),
    )
    assert id_a == id_b


# requested_window carries time into identity.


class TestRequestedWindow:
    _SLOTS: ClassVar[list[Slot]] = [_value_slot("what", "SomeProp", "SomeValue")]

    def test_none_maps_to_all_distinct_from_latest(self) -> None:
        id_none = compute_spec_id(
            shape_for("shape"), self._SLOTS, requested_window=None, open_roles=()
        )
        id_latest = compute_spec_id(
            shape_for("shape"), self._SLOTS, requested_window="latest", open_roles=()
        )
        assert id_none != id_latest

    @pytest.mark.parametrize(
        ("window_a", "window_b"),
        [
            pytest.param(None, TimeWindow(start_year=2015, end_year=2020), id="all-vs-range"),
            pytest.param(
                "latest", TimeWindow(start_year=2015, end_year=2020), id="latest-vs-range"
            ),
            pytest.param(
                TimeWindow(start_year=2015, end_year=2020),
                TimeWindow(end_year=2020),
                id="full-range-vs-open-start",
            ),
            pytest.param(
                TimeWindow(start_year=2015, end_year=2020),
                TimeWindow(start_year=2015),
                id="full-range-vs-open-end",
            ),
            pytest.param(
                TimeWindow(end_year=2020),
                TimeWindow(start_year=2020),
                id="open-start-vs-open-end",
            ),
            pytest.param(
                TimeWindow(start_year=2010, end_year=2020),
                TimeWindow(start_year=2005, end_year=2020),
                id="different-bounds",
            ),
            pytest.param(
                TimeWindow(start_year=2010, end_year=2020),
                TimeWindow(start_year=2005, end_year=2020),
                id="when-slot-companion",
            ),
        ],
    )
    def test_window_shapes_remain_distinct(
        self,
        window_a: TimeWindow | str | None,
        window_b: TimeWindow | str | None,
    ) -> None:
        assert compute_spec_id(
            shape_for("shape"), self._SLOTS, requested_window=window_a, open_roles=()
        ) != compute_spec_id(
            shape_for("shape"), self._SLOTS, requested_window=window_b, open_roles=()
        )


def test_reversed_requested_window_raises_identity_error() -> None:
    """The adapter raises IdentityError for reversed bounds; request doors reject them earlier.

    TimeWindow permits a reversed range, so identity validation catches it here.
    """
    slots = [_value_slot("what", "SomeProp", "SomeValue")]
    with pytest.raises(IdentityError) as exc_info:
        compute_spec_id(
            shape_for("shape"),
            slots,
            requested_window=TimeWindow(start_year=2020, end_year=2015),
            open_roles=(),
        )
    assert exc_info.value.category == "reversed_range"


def test_mid_stream_and_final_ids_agree_despite_differing_when_slot_binding() -> None:
    """Sentence events and final specs can carry different when slots yet share an id.

    Both use the same requested_window, the time field hashed by spec1.
    """
    requested = TimeWindow(start_year=2010, end_year=2020)
    coverage = TimeWindow(start_year=2015, end_year=2018)
    entities = [_subject_entity("country/IND")]
    slots = [_value_slot("how", "activitySource", "GrossDomesticProduction")]

    mid_stream_id = compute_spec_id(
        shape_for("shape"),
        [*slots, _when_slot_value(requested.start_year, requested.end_year)],
        entities=entities,
        requested_window=requested,
        open_roles=(),
    )
    final_id = compute_spec_id(
        shape_for("shape"),
        [*slots, _when_slot_value(coverage.start_year, coverage.end_year)],
        entities=entities,
        requested_window=requested,
        open_roles=(),
    )
    assert mid_stream_id == final_id


class TestMidStreamAndFinalIdsAgreeAcrossPlaceConstraintModes:
    """End-to-end calls verify sentence and final spec ids match in both modes.

    The test catches argument errors at resolve_events call sites.
    """

    _QUERY = "fertility rate in Kenya"

    @staticmethod
    def _stream_place_constraint_off(
        query: str, search_settings, place_constraint_off: bool = True
    ) -> tuple[list[str], str]:
        """Return sentence and final ids, setting place-as-constraint through seam_default."""
        import asyncio

        from search.engine.core import resolve_events
        from search.models import RawTextInput, ResolveRequest
        from tests.engine._fake_sdmx import FakeSdmxClient
        from tests.fixtures import FakeGraph, FakeLLM

        async def _body() -> tuple[list[str], str]:
            sentence_ids: list[str] = []
            final_id = ""
            async for ev in resolve_events(
                ResolveRequest(input=RawTextInput(query=query)),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(),
            ):
                if ev.kind == "sentence":
                    sentence_ids.append(ev.spec_id)
                elif ev.kind == "response":
                    final_id = ev.response.parts[0].spec.spec_id
            return sentence_ids, final_id

        search_settings(seam_default=not place_constraint_off)
        return asyncio.run(_body())

    def test_mid_stream_id_matches_the_final_spec(self, search_settings) -> None:
        sentence_ids, final_id = self._stream_place_constraint_off(self._QUERY, search_settings)

        assert sentence_ids, "expected at least one sentence event"
        assert final_id, "expected a definite response carrying a Spec"
        assert sentence_ids[0] == final_id, (
            "the FIRST sentence event's spec_id must equal the final Spec's -- "
            "a consumer joining on the mid-stream id joins nothing otherwise"
        )
        assert len(set(sentence_ids)) == 1, (
            f"every sentence event for one variable must carry one id, got {sentence_ids}"
        )

    def test_place_constraint_on_is_unaffected(self, search_settings) -> None:
        """Place-as-constraint mode also yields matching event and spec ids."""
        sentence_ids, final_id = self._stream_place_constraint_off(
            self._QUERY, search_settings, place_constraint_off=False
        )

        assert sentence_ids and final_id
        assert sentence_ids[0] == final_id
