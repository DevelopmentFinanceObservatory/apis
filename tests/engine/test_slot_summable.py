"""Test `Slot.summable`: a slot sums when its code system is declared summable and the
dataset elected for the family reads an absent observation as zero."""

from __future__ import annotations

import dataclasses

import pytest

import search.engine.retrieval.bound_values as bound_values_module
from code_index import CodeIndex
from code_index.shared import get_index
from code_index.types import StatVarRow
from datasets_api.groupby import collapse_axes
from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.bound_values import (
    apply_slot_expansions,
    apply_slot_summability,
    identity_dataset,
    slot_expansions,
)
from search.engine.spec_build import build_spec, identity_shape_from_wire
from search.engine.spec_id import build_spec_identity
from search.models import (
    BaseIdentity,
    BindingUnbound,
    CoverageBare,
    Slot,
    SlotKey,
    StatVar,
)
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_PROVENANCE,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)
from tests.engine._builders import (
    binding_set_slot,
    gref,
    shape_five_tuple,
    unbound_slot,
    value_slot,
)

_HEALTH = "CRSPurposeCode120"  # hierarchical crsPurpose parent
_EDUCATION = "CRSPurposeCode110"
_SYNTHETIC_NOT_SUMMABLE = "syntheticNotSummable"

_CRS_AXES = ("crsCooperationModality", "crsPurpose", "crsTypeOfFinance", "crsTypeOfFlow")
# The imputed five-tuple declares two of the CRS code systems.
_IMPUTED_CRS_AXES = ("crsPurpose", "crsTypeOfFlow")


def _summable_by_property(slots: list[Slot]) -> dict[str, bool]:
    return {
        slot.key.property.ref if slot.key.property else slot.key.label: slot.summable
        for slot in slots
    }


def _identity(document: dict) -> BaseIdentity:
    return BaseIdentity.model_validate(document)


def _slots_of(document: dict) -> list[Slot]:
    """One wire slot per identity slot, with the identity's own binding."""
    slots: list[Slot] = []
    for slot in document["slots"]:
        prop, binding = slot["property"], slot["binding"]
        if binding["state"] == "value":
            slots.append(value_slot("how", prop, prop, binding["ref"]))
        else:
            slots.append(unbound_slot("how", prop, prop))
    return slots


def _unindexed_identity() -> BaseIdentity:
    """A described identity whose five-tuple the packaged index does not carry."""
    return _identity(
        {
            "identity_version": "spec1",
            "shape": {
                "kind": "statvar_schema",
                "population_type": "EconomicActivity",
                "measured_property": "amount",
                "stat_type": "measuredValue",
            },
            "slots": [{"property": "crsPurpose", "binding": {"state": "unbound"}}],
            "requested_window": {"kind": "all"},
        }
    )


def _index_with(index: CodeIndex, **changes) -> CodeIndex:
    """A copy of `index` with the given constructor arguments replaced."""
    parts = {
        "stamp": index.stamp,
        "declarations": tuple(index.declarations.values()),
        "codes": index.codes,
        "statvars": index.statvars,
        "datasets": tuple(index.datasets.values()),
    }
    return CodeIndex(**{**parts, **changes})


class TestApplySlotSummability:
    @pytest.mark.parametrize(
        "slot",
        [
            unbound_slot("how", "crsPurpose", "crsPurpose"),
            binding_set_slot("how", "crsPurpose", "crsPurpose", [_HEALTH, _EDUCATION]),
            value_slot("how", "crsPurpose", "crsPurpose", _HEALTH),
        ],
        ids=["unbound", "set", "value"],
    )
    def test_a_crs_purpose_slot_of_a_crs_family_reads_summable_whatever_its_binding(self, slot):
        identity = _identity(education_aid_identity())
        assert identity_dataset(identity) == CRS_PROVENANCE
        (stamped,) = apply_slot_summability([slot], identity)
        assert stamped.summable is True

    @pytest.mark.parametrize(
        "slot",
        [
            value_slot("what", "gender", "gender", "Female"),
            unbound_slot("how", "measurementMethod", "measurementMethod"),
            Slot(key=SlotKey(axis="where", property=None, label="where"), binding=BindingUnbound()),
        ],
        ids=["gender", "measurement", "property-less"],
    )
    def test_a_slot_off_any_code_system_reads_not_summable(self, slot):
        (stamped,) = apply_slot_summability([slot], _identity(education_aid_identity()))
        assert stamped.summable is False

    def test_the_aid_type_slot_reads_not_summable(self):
        document = imputed_education_aid_identity()
        flags = _summable_by_property(
            apply_slot_summability(_slots_of(document), _identity(document))
        )
        assert flags[AID_TYPE] is False
        assert {axis: flags[axis] for axis in flags if axis in _CRS_AXES} == dict.fromkeys(
            _IMPUTED_CRS_AXES, True
        )

    def test_an_identity_leaving_a_pinned_slot_unbound_reads_the_bound_identity_s_flags(self):
        """The aid type is implied by the election, so an identity leaving it unbound has the
        family, and each slot the summability, of the identity binding it."""
        bound = imputed_education_aid_identity()
        document = imputed_education_aid_identity()
        aid_type = next(slot for slot in document["slots"] if slot["property"] == AID_TYPE)
        aid_type["binding"] = {"state": "unbound"}
        flags = _summable_by_property(
            apply_slot_summability(_slots_of(document), _identity(document))
        )
        assert flags == _summable_by_property(
            apply_slot_summability(_slots_of(bound), _identity(bound))
        )

    def test_a_family_the_index_does_not_resolve_reads_not_summable(self):
        identity = _unindexed_identity()
        assert identity_dataset(identity) is None
        (stamped,) = apply_slot_summability(
            [unbound_slot("how", "crsPurpose", "crsPurpose")], identity
        )
        assert stamped.summable is False

    def test_a_family_of_a_dataset_with_no_absence_reading_reads_not_summable(self, monkeypatch):
        index = get_index()
        edited = _index_with(
            index,
            datasets=tuple(
                dataclasses.replace(dataset, absent_observation="unknown")
                if dataset.provenance == IMPUTED_PROVENANCE
                else dataset
                for dataset in index.datasets.values()
            ),
        )
        monkeypatch.setattr(bound_values_module, "get_index", lambda: edited)
        imputed = imputed_education_aid_identity()
        crs = education_aid_identity()

        imputed_flags = _summable_by_property(
            apply_slot_summability(_slots_of(imputed), _identity(imputed))
        )
        crs_flags = _summable_by_property(apply_slot_summability(_slots_of(crs), _identity(crs)))

        assert set(imputed_flags.values()) == {False}
        assert {crs_flags[axis] for axis in _CRS_AXES} == {True}

    def test_stamping_keeps_order_binding_and_expansion(self):
        crs = value_slot("how", "crsPurpose", "crsPurpose", _HEALTH)
        expanded = apply_slot_expansions(
            [crs],
            slot_expansions(
                [
                    SlotBindingDraft(
                        axis="how", property_dcid="crsPurpose", kind="value", value_dcids=[_HEALTH]
                    )
                ]
            ),
        )
        gender = value_slot("what", "gender", "gender", "Female")

        stamped = apply_slot_summability([gender, *expanded], _identity(education_aid_identity()))

        assert [s.key for s in stamped] == [gender.key, crs.key]
        assert [s.binding for s in stamped] == [gender.binding, crs.binding]
        assert stamped[1].expansion == expanded[0].expansion is not None
        assert [s.summable for s in stamped] == [False, True]


class TestAgreesWithGroupby:
    """In a family whose dataset reads absence as zero, a slot reads summable exactly when
    groupby collapses its code system."""

    def test_every_declared_code_system_agrees(self, monkeypatch):
        index = get_index()
        template = next(row for row in index.statvars if row.provenances == (CRS_PROVENANCE,))
        synthetic = dataclasses.replace(
            index.declarations["crsPurpose"],
            code_system=_SYNTHETIC_NOT_SUMMABLE,
            hierarchical=False,
            summable=False,
            default=None,
        )
        edited = _index_with(
            index,
            declarations=(*index.declarations.values(), synthetic),
            statvars=(
                *index.statvars,
                StatVarRow(
                    dcid="SyntheticNotSummableMember",
                    population_type=template.population_type,
                    measured_property=template.measured_property,
                    stat_type=template.stat_type,
                    measurement_qualifier=template.measurement_qualifier,
                    measurement_denominator=template.measurement_denominator,
                    observation_properties=template.observation_properties,
                    constraint_properties={
                        **template.constraint_properties,
                        _SYNTHETIC_NOT_SUMMABLE: "SyntheticValue",
                    },
                    provenances=template.provenances,
                ),
            ),
        )
        monkeypatch.setattr(bound_values_module, "get_index", lambda: edited)
        declarations = edited.declarations
        document = education_aid_identity()
        document["slots"].append(
            {"property": _SYNTHETIC_NOT_SUMMABLE, "binding": {"state": "unbound"}}
        )

        flags = _summable_by_property(
            apply_slot_summability(_slots_of(document), _identity(document))
        )

        groupby_collapses = {
            code_system: bool(
                collapse_axes(by=[], decoded_code_systems=[code_system], declarations=declarations)
            )
            for code_system in declarations
        }
        declared_slots = {*_CRS_AXES, _SYNTHETIC_NOT_SUMMABLE}
        assert set(flags) & set(declarations) == declared_slots
        assert {code_system: flags[code_system] for code_system in declared_slots} == {
            code_system: groupby_collapses[code_system] for code_system in declared_slots
        }
        assert flags[_SYNTHETIC_NOT_SUMMABLE] is False
        assert flags["crsPurpose"] is True

    def test_the_packaged_code_systems_read_their_declared_summability(self):
        flags = {
            prop: summable
            for document in (education_aid_identity(), imputed_education_aid_identity())
            for prop, summable in _summable_by_property(
                apply_slot_summability(_slots_of(document), _identity(document))
            ).items()
        }
        assert {
            code_system: flags[code_system]
            for code_system in get_index().declarations
            if code_system in flags
        } == {
            "crsPurpose": True,
            "crsTypeOfFinance": True,
            "crsCooperationModality": True,
            "crsTypeOfFlow": True,
            "dac2aAidType": False,
        }


class TestBuildSpec:
    def _spec(self, slots: list[Slot], *, pop: str = "EconomicActivity", mq: str | None = None):
        shape = shape_five_tuple(pop, "amount", mq)
        return build_spec(
            open_roles=(),
            shape=shape,
            slots=slots,
            stat_vars=[
                StatVar(
                    ref=gref("test_sv"),
                    shape_id=shape.shape_id,
                    slot_values=[],
                    constraint_properties=[],
                )
            ],
            entities=[],
            coverage=CoverageBare(has_data=True),
            measure_elected_properties=frozenset(),
        )

    def test_served_slots_of_an_index_resolved_family_carry_the_flag(self):
        document = education_aid_identity()
        spec = self._spec(_slots_of(document), pop="FinancialFlow", mq="Nominal")

        assert spec.definition.identity.slots == _identity(document).slots
        flags = _summable_by_property(spec.resolved.slots)
        assert {flags[axis] for axis in _CRS_AXES} == {True}
        assert flags["financialFlowType"] is False

    def test_served_slots_of_a_family_outside_the_index_read_not_summable(self):
        spec = self._spec(
            [
                value_slot("how", "crsPurpose", "crsPurpose", _HEALTH),
                value_slot("what", "gender", "gender", "Female"),
            ]
        )
        assert _summable_by_property(spec.resolved.slots) == {"crsPurpose": False, "gender": False}

    def test_the_flag_leaves_spec_id_and_identity_unchanged(self):
        slots = [
            binding_set_slot("how", "crsPurpose", "crsPurpose", [_HEALTH, _EDUCATION]),
            value_slot("what", "gender", "gender", "Female"),
        ]
        shape = identity_shape_from_wire(shape_five_tuple("EconomicActivity", "amount"))
        unstamped = build_spec_identity(shape, slots, entities=[], open_roles=())
        stamped = build_spec_identity(
            shape,
            [s.model_copy(update={"summable": True}) for s in slots],
            entities=[],
            open_roles=(),
        )

        assert stamped == unstamped
        assert self._spec(slots).spec_id == unstamped[0]
        assert "summable" not in unstamped[1].model_dump_json()
