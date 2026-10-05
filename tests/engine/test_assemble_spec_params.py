"""Test build_spec date metadata, facets, and identity hashing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.engine.spec_build import build_spec
from search.models import (
    BreadthDim,
    CoverageExact,
    GraphRef,
    ResolvedFacet,
    StatVar,
    TimeWindow,
)
from tests.engine._assemble_fixtures import (
    _directional_entity,
    _make_coverage,
    _make_entity,
    _make_five_tuple_refs,
    _make_pipeline_steps,
    _make_spec,
    _standard_shape_draft,
)

# applied_window / date_source in ResolutionTrace


def _make_exact_coverage(window: TimeWindow | None) -> CoverageExact:
    return CoverageExact(
        has_data=True,
        observation_count=42,
        dimensions=[
            BreadthDim(label="sources", count=1),
            BreadthDim(label="observations", count=42),
        ],
        window=window,
    )


def test_build_spec_applied_window_and_date_source_when_window_present():
    window = TimeWindow(start_year=2015, end_year=2020)
    from search.engine.bind import SlotBindingDraft
    from search.engine.spec_build import build_shape_model, build_slot, build_spec

    shape_draft = _standard_shape_draft()
    five_tuple_refs = _make_five_tuple_refs()

    scheme_draft = SlotBindingDraft(
        axis="what",
        property_dcid="TestSchemeProp",
        kind="value",
        value_dcids=["ODAGrants"],
    )
    purpose_draft = SlotBindingDraft(
        axis="how",
        property_dcid="TestPurposeProp",
        kind="value",
        value_dcids=["DAC/Health"],
    )
    place_draft = SlotBindingDraft(
        axis="where",
        property_dcid="TestPlaceProp",
        kind="value",
        value_dcids=["country/ETH"],
    )

    slots = []
    slot_key_models = []
    for slot_draft_key, binding_draft, grounded in [
        (shape_draft.slot_keys[0], scheme_draft, [GraphRef(dcid="ODAGrants", label="ODA Grants")]),
        (shape_draft.slot_keys[1], purpose_draft, [GraphRef(dcid="DAC/Health", label="Health")]),
        (
            shape_draft.slot_keys[2],
            place_draft,
            [GraphRef(dcid="country/ETH", label="Ethiopia")],
        ),
    ]:
        prop_dcid = slot_draft_key.property_dcid
        assert prop_dcid is not None
        prop_ref = GraphRef(dcid=prop_dcid, label=prop_dcid)
        slot = build_slot(slot_draft_key, binding_draft, grounded, property_ref=prop_ref)
        slots.append(slot)
        slot_key_models.append(slot.key)

    shape_model = build_shape_model(
        shape_draft, slot_key_models, five_tuple_refs, member_count=1, members=None
    )
    sv_ref = GraphRef(dcid="Count_Person", label="Health ODA")
    stat_vars = [
        StatVar(ref=sv_ref, shape_id=shape_draft.shape_id, slot_values=[], constraint_properties=[])
    ]

    result_spec = build_spec(
        measure_elected_properties=frozenset(),
        shape=shape_model,
        slots=slots,
        stat_vars=stat_vars,
        entities=[_make_entity("country/ETH", "Ethiopia")],
        coverage=_make_exact_coverage(window),
        pipeline_trace=_make_pipeline_steps(),
        date_source="query",
    )

    assert result_spec.resolved.trace.applied_window == window
    assert result_spec.resolved.trace.date_source == "query"


def test_build_spec_applied_window_and_date_source_none_when_no_window():
    spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    assert spec.resolved.trace.applied_window is None
    assert spec.resolved.trace.date_source is None


# build_spec date_source explicit override


def test_build_spec_date_source_explicit_coverage_clamp():
    window = TimeWindow(start_year=2015, end_year=2020)
    spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    sv_ref = GraphRef(dcid="Count_Person", label="Health ODA")
    result = build_spec(
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[
            StatVar(
                ref=sv_ref,
                shape_id=spec.resolved.shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=_make_exact_coverage(window),
        pipeline_trace=[],
        date_source="coverage_clamp",
    )
    assert result.resolved.trace.date_source == "coverage_clamp"
    assert result.resolved.trace.applied_window == window


def test_build_spec_date_source_default_infers_query_when_window():
    window = TimeWindow(start_year=2018)
    spec = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    result = build_spec(
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[],
        entities=[],
        coverage=_make_exact_coverage(window),
        pipeline_trace=[],
    )
    assert result.resolved.trace.date_source == "query"


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
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[],
        entities=[],
        coverage=_make_coverage(),
        pipeline_trace=_make_pipeline_steps(),
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
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[],
        entities=[],
        coverage=_make_coverage(),
        pipeline_trace=_make_pipeline_steps(),
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
            measure_elected_properties=frozenset(),
            shape=spec.resolved.shape,
            slots=[],
            stat_vars=[],
            entities=[],
            coverage=_make_coverage(),
            pipeline_trace=_make_pipeline_steps(),
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
