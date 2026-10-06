"""Test build_spec date metadata, facets, and identity hashing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.engine.spec_build import build_spec
from search.models import (
    ResolvedFacet,
)
from tests.engine._assemble_fixtures import (
    _directional_entity,
    _make_coverage,
    _make_entity,
    _make_spec,
)

# Facet variants survive JSON serialisation.


def test_build_spec_facets_serialised_body_has_the_four_resolved_facet_shapes():
    spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    facets = {
        "measurementMethod": ResolvedFacet(
            basis="bound", ref="GrossDisbursement", label="Gross Disbursement"
        ),
        "provenance": ResolvedFacet(basis="fallback", selection="per_series"),
        "unit": ResolvedFacet(
            basis="bound",
            ref="EUR",
            label="Euro",
            request_basis={"currency": "bound", "prices": "bound"},
        ),
    }
    result = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[],
        entities=[],
        coverage=_make_coverage(),
        facets=facets,
    )
    body = result.model_dump(mode="json", exclude_none=True)
    resolved_facets = body["resolved"]["facets"]

    # Bound reference omits selection and request basis.
    assert resolved_facets["measurementMethod"] == {
        "basis": "bound",
        "ref": "GrossDisbursement",
        "label": "Gross Disbursement",
    }
    # Per-series provenance uses selection.
    assert resolved_facets["provenance"] == {"basis": "fallback", "selection": "per_series"}
    # Unit facets can carry the request basis.
    assert resolved_facets["unit"] == {
        "basis": "bound",
        "ref": "EUR",
        "label": "Euro",
        "request_basis": {"currency": "bound", "prices": "bound"},
    }
    # Only supplied axes appear.
    assert set(resolved_facets) == {"measurementMethod", "provenance", "unit"}


def test_build_spec_facets_defaults_to_empty_when_not_supplied():
    spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    result = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[],
        entities=[],
        coverage=_make_coverage(),
    )
    assert result.resolved.facets == {}
    body = result.model_dump(mode="json", exclude_none=True)
    assert body["resolved"]["facets"] == {}


def test_build_spec_rejects_request_basis_on_a_non_unit_axis():
    """request_basis is valid only on the unit axis."""
    spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    facets = {
        "measurementMethod": ResolvedFacet(
            basis="bound",
            ref="GrossDisbursement",
            request_basis={"currency": "bound"},
        ),
    }
    with pytest.raises(ValidationError, match="request_basis is only meaningful"):
        build_spec(
            open_roles=(),
            measure_elected_properties=frozenset(),
            shape=spec.resolved.shape,
            slots=[],
            stat_vars=[],
            entities=[],
            coverage=_make_coverage(),
            facets=facets,
        )


# Hash directional identity separately from presented entities.


def test_build_spec_identity_entities_none_hashes_entities_itself():
    """By default, the hash uses the presented entities."""
    entities = [_make_entity("country/ETH", "Ethiopia")]
    with_default = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        entities=entities,
    )
    explicit_same = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        entities=entities,
        identity_entities=entities,
    )
    assert with_default.spec_id == explicit_same.spec_id


def test_build_spec_identity_entities_overrides_hash_not_presentation():
    """identity_entities changes the hash while entities stay unchanged."""
    presented = [_make_entity("country/ETH", "Ethiopia")]
    hashed_as_directional = [_directional_entity("country/ETH", "destination")]

    subject_spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        entities=presented,
    )
    overridden_spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        entities=presented,
        identity_entities=hashed_as_directional,
    )

    assert overridden_spec.resolved.entities == subject_spec.resolved.entities
    assert all(e.role == "observationAbout" for e in overridden_spec.resolved.entities)
    assert overridden_spec.spec_id != subject_spec.spec_id


def test_build_spec_place_constraint_off_directional_collision_fixed():
    """Reversed donor-recipient directions must hash to different spec ids."""
    presented = [
        _make_entity("country/USA", "United States"),
        _make_entity("country/ETH", "Ethiopia"),
    ]

    usa_to_eth = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        entities=presented,
        identity_entities=[
            _directional_entity("country/USA", "origin"),
            _directional_entity("country/ETH", "destination"),
        ],
    )
    eth_to_usa = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        entities=presented,
        identity_entities=[
            _directional_entity("country/ETH", "origin"),
            _directional_entity("country/USA", "destination"),
        ],
    )

    # Both directions have the same presented subject entities.
    assert usa_to_eth.resolved.entities == eth_to_usa.resolved.entities
    assert all(e.role == "observationAbout" for e in usa_to_eth.resolved.entities)
    # Direction enters the identity hash.
    assert usa_to_eth.spec_id != eth_to_usa.spec_id
