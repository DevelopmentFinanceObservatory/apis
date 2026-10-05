"""Checks precedence, duplicate suppression, and token metadata for measure labels."""

from __future__ import annotations

import pytest

from search.models import (
    BindingValue,
    IdentityFacetBinding,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    ResolvedFacet,
    Slot,
    SlotValue,
)
from search.render.plan import render_sentence_plan
from tests.render._helpers import (
    _ref,
    _rendered,
    _slot_key,
    _spec,
)


def _flow_type_slot(label: str) -> Slot:
    """Build a bound how slot for the financial-flow measure axis."""
    return Slot(
        key=_slot_key("how", "financialFlowType", "CRS Flow Type", "flow type"),
        binding=BindingValue(
            value=SlotValue(ref=_ref("Disbursement", label), value_kind="enum_value")
        ),
    )


def _netting_basis_slot(label: str) -> Slot:
    return Slot(
        key=_slot_key("how", "nettingBasis", "CRS Netting Basis", "netting basis"),
        binding=BindingValue(value=SlotValue(ref=_ref("Gross", label), value_kind="enum_value")),
    )


def _purpose_slot(label: str) -> Slot:
    return Slot(
        key=_slot_key("how", "crsPurpose", "CRS Purpose", "purpose"),
        binding=BindingValue(
            value=SlotValue(ref=_ref("MalariaControl", label), value_kind="enum_value")
        ),
    )


def test_measure_label_replaces_the_composed_what_phrase():
    """Without a bound what slot, use the measure label as the sentence subject."""
    spec = _spec(measure_label="gross disbursements")
    assert _rendered(spec) == "Gross disbursements."


def test_measure_label_suppresses_measure_extra_tokens():
    """A measure label replaces the separate stat-type and qualifier tokens."""
    spec = _spec(
        measurement_qualifier=_ref("Nominal", "Nominal"),
        measure_label="gross disbursements",
    )
    assert _rendered(spec) == "Gross disbursements."


def test_measure_label_axes_suppress_the_matching_bound_how_slots():
    """Measure axes stay empty; the ordinary purpose slot still renders."""
    spec = _spec(
        slots=[
            _flow_type_slot("disbursement"),
            _netting_basis_slot("gross"),
            _purpose_slot("malaria control"),
        ],
        measure_label="gross disbursements",
        measure_label_axes=["financialFlowType", "nettingBasis"],
    )
    rendered = _rendered(spec)
    assert rendered == "Gross disbursements for malaria control."
    assert "disbursement " not in rendered.lower()  # the suppressed how-clause text
    assert "gross " not in rendered.lower() or rendered.lower().startswith("gross ")


def test_measure_suppressed_slot_keeps_its_chip_backing():
    """A suppressed measure axis retains its token metadata."""
    slot = _flow_type_slot("disbursement")
    spec = _spec(
        slots=[slot],
        measure_label="gross disbursements",
        measure_label_axes=["financialFlowType"],
    )
    plan = render_sentence_plan(spec)
    how_tokens = [t for t in plan if t.axis == "how"]
    assert len(how_tokens) == 1
    token = how_tokens[0]
    assert token.text == ""
    assert token.property_dcid == "financialFlowType"
    assert token.binding_ref == slot.key.as_ref()


def test_unlabelled_path_is_unchanged_by_measure_label_axes_alone():
    """Without a measure label, property-based wording, a qualifier that is no price
    basis, and how slots render normally."""
    spec = _spec(
        measurement_qualifier=_ref("AgeAdjusted", "Age adjusted"),
        slots=[_flow_type_slot("disbursement")],
    )
    rendered = _rendered(spec)
    assert "age adjusted" in rendered.lower()
    assert "disbursement" in rendered.lower()


def test_measure_label_used_with_a_bound_what_slot_defers_to_it():
    """A bound what slot takes precedence over `measure_label`."""
    from tests.render._helpers import _what_slot

    spec = _spec(
        slots=[_what_slot("ODA grants")],
        measure_label="gross disbursements",
    )
    assert _rendered(spec) == "ODA grants."


def _euro_facets() -> dict[str, ResolvedFacet]:
    return {"unit": ResolvedFacet(basis="bound", ref="CurrencyEUR", label="Euro")}


def _unit_request(**request) -> dict[str, IdentityFacetBinding]:
    return {"unit": IdentityFacetValueRequest(request=IdentityUnitRequest(**request))}


@pytest.mark.parametrize(
    ("measurement_qualifier", "facets", "identity_facets"),
    [
        pytest.param(
            None,
            _euro_facets(),
            _unit_request(currency="EUR", prices="constant", base_year=2024),
            id="constant-request",
        ),
        pytest.param(
            None, _euro_facets(), _unit_request(currency="EUR", prices="current"), id="current"
        ),
        pytest.param(
            _ref("Nominal", "Nominal"),
            _euro_facets(),
            _unit_request(currency="EUR", prices="constant", base_year=2024),
            id="request-over-native-qualifier",
        ),
        pytest.param(
            None,
            {"unit": ResolvedFacet(basis="bound", selection="per_series")},
            _unit_request(currency="EUR", prices="constant", base_year=2024),
            id="per-series-currency",
        ),
        pytest.param(_ref("Nominal", "Nominal"), {}, None, id="native-nominal"),
        pytest.param(_ref("RealValue", "Real value"), {}, None, id="native-real"),
    ],
)
def test_the_heading_never_states_a_price_basis(measurement_qualifier, facets, identity_facets):
    """Current or constant prices belong to the unit, so neither a unit request nor a
    price-basis qualifier reaches the heading."""
    spec = _spec(
        measure_label="gross disbursements",
        measurement_qualifier=measurement_qualifier,
        facets=facets,
        identity_facets=identity_facets,
    )
    assert _rendered(spec) == "Gross disbursements."
