"""Pins entity roles and canonical provenance for `identity_edit.synthesize_base`."""

from __future__ import annotations

import re

import pytest

from search.engine.identity_edit import synthesize_base
from search.models import (
    BindingValue,
    GraphRef,
    IdentityBindingValue,
    Slot,
    SlotKey,
    SlotValue,
)

_COUNT_PERSON_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "count",
    "stat_type": "measuredValue",
}


def test_synthesize_base_stamps_entity_role_subject():
    """Subject entities map to the identity key `observationAbout`."""
    _base_spec_id, base_identity = synthesize_base(
        _COUNT_PERSON_SHAPE,
        slots=[],
        entity_dcids=["country/KEN"],
        resolved_stat_var_dcids=None,
    )
    assert base_identity is not None
    assert base_identity.entities == {"observationAbout": ["country/KEN"]}


def test_synthesize_base_stamps_entity_role_directional_for_role_bearing_shape():
    """Role-bearing shapes map directional entities to the identity key `destination`."""
    _base_spec_id, base_identity = synthesize_base(
        _COUNT_PERSON_SHAPE,
        slots=[],
        entity_dcids=["country/KEN"],
        resolved_stat_var_dcids=None,
        declared_roles=("creditor", "debtor"),
    )
    assert base_identity is not None
    assert base_identity.entities == {"destination": ["country/KEN"]}


def _source_slot(dcid: str) -> Slot:
    return Slot(
        key=SlotKey(axis="source", property=None, label="data source"),
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(dcid=dcid, label=dcid), value_kind="source")
        ),
    )


def test_synthesize_base_preserves_legacy_provenance_via_source_slot():
    """Legacy provenance and a source slot produce the same identity and retain the facets."""
    facets = {
        "provenance": IdentityBindingValue(ref="dc/base/WorldDevelopmentIndicators"),
        "measurementMethod": IdentityBindingValue(ref="dc/measurementMethod/MethodA"),
        "unit": IdentityBindingValue(ref="Count_Person"),
    }
    legacy_spec_id, legacy_identity = synthesize_base(
        _COUNT_PERSON_SHAPE,
        slots=[],
        entity_dcids=["country/KEN"],
        resolved_stat_var_dcids=None,
        facets=facets,
    )
    source_spec_id, source_identity = synthesize_base(
        _COUNT_PERSON_SHAPE,
        slots=[_source_slot("dc/base/WorldDevelopmentIndicators")],
        entity_dcids=["country/KEN"],
        resolved_stat_var_dcids=None,
        facets={key: value for key, value in facets.items() if key != "provenance"},
    )

    assert legacy_spec_id == source_spec_id
    assert legacy_identity == source_identity
    assert legacy_identity is not None
    assert legacy_identity.facets == facets


def test_synthesize_base_refuses_legacy_source_slot_provenance_conflict():
    with pytest.raises(ValueError, match=re.escape("source-axis slot and facets.provenance")):
        synthesize_base(
            _COUNT_PERSON_SHAPE,
            slots=[_source_slot("dc/base/SomeSource")],
            entity_dcids=["country/KEN"],
            resolved_stat_var_dcids=None,
            facets={"provenance": IdentityBindingValue(ref="dc/base/OtherSource")},
        )
