"""Test unsupported-definition projection and response assembly."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.engine.assemble import build_unsupported_part
from search.engine.spec_build import build_unsupported_definition
from search.models import (
    SCHEMA_VERSION,
    BindingValue,
    Entity,
    EntityRoleSubject,
    GraphRef,
    Slot,
    SlotKey,
    SlotValue,
)

# build_unsupported_definition / build_unsupported_part


def test_build_unsupported_definition_projects_identity_without_spec_id():
    """The unsupported definition carries identity fields but no spec id."""
    slots = [
        Slot(
            key=SlotKey(
                axis="what", property=GraphRef(dcid="propA", label="Prop A"), label="Prop A"
            ),
            binding=BindingValue(
                value=SlotValue(ref=GraphRef(dcid="valA", label="Val A"), value_kind="enum_value")
            ),
        ),
    ]
    entities = [
        Entity(
            ref=GraphRef(dcid="country/KEN", label="Kenya"),
            entity_type=None,
            role=EntityRoleSubject(),
        )
    ]

    shape = {"kind": "statvar_schema", "population_type": "some_shape_id"}
    definition = build_unsupported_definition(
        shape, slots, entities=entities, requested_window=None
    )

    assert definition["schema_version"] == SCHEMA_VERSION
    assert "spec_id" not in definition
    identity = definition["identity"]
    assert "spec_id" not in identity
    assert identity["identity_version"] == "spec1"
    assert identity["shape"] == shape
    assert identity["slots"] == [
        {"property": "propA", "binding": {"state": "value", "ref": "valA"}}
    ]
    assert identity["entities"] == {"observationAbout": ["country/KEN"]}
    assert identity["requested_window"] == {"kind": "all"}


def test_build_unsupported_definition_rejects_a_zero_member_shape():
    """A described identity without schema members fails validation."""
    with pytest.raises(ValidationError):
        build_unsupported_definition(
            {"kind": "statvar_schema"},
            [],
            entities=None,
            requested_window=None,
        )


def test_build_unsupported_part_wraps_definition_into_part():
    definition = build_unsupported_definition(
        {"kind": "statvar_schema", "population_type": "shape_x"},
        [],
        entities=None,
        requested_window=None,
    )

    part = build_unsupported_part(
        "unit_request_unsupported",
        "/identity/facets/unit",
        "a PPP-terms unit request is not expressible at this schema version",
        definition,
        variable_text="price of X in PPP terms",
        warnings=[],
    )

    assert part.status == "unsupported"
    assert part.unsupported.code == "unit_request_unsupported"
    assert part.unsupported.pointer == "/identity/facets/unit"
    assert (
        part.unsupported.detail
        == "a PPP-terms unit request is not expressible at this schema version"
    )
    assert part.unsupported.definition == definition
