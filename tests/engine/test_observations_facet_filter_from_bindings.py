"""Tests for translating posted constraints into facet filters."""

from __future__ import annotations

from search.engine.clients.graph import Facet
from search.engine.region.resubmit_request import _facet_filter_from_bindings
from search.models import (
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
)
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# region/resubmit_request.py: _facet_filter_from_bindings -- the posted-constraint ->
# facet_filter builder region/resubmit_promote.py's own call site uses.
# ---------------------------------------------------------------------------

# unit dcid -> ISO 4217 alpha-3, standing in for this deployment's graph
# (search.engine.unit.refs reads it live via graph.currency_units()).
_GRAPH = FakeGraph(currency_units={"USD": "USD", "EUR": "EUR", "GBP": "GBP"})


def _facet(*, measurement_method: str | None = None, unit: str | None = None, provenance_id=None):

    return Facet(
        earliest_date=None,
        latest_date=None,
        obs_count=0,
        provenance_id=provenance_id,
        measurement_method=measurement_method,
        unit=unit,
    )


def test_facet_filter_from_bindings_none_when_nothing_posted():
    assert _facet_filter_from_bindings(_GRAPH, [], None, deadline=None) is None
    assert _facet_filter_from_bindings(_GRAPH, [], {}, deadline=None) is None


def test_facet_filter_from_bindings_provenance_only_from_source_dcids():
    """Provenance comes from the source-axis slot."""
    result = _facet_filter_from_bindings(
        _GRAPH,
        ["dc/base/allowed"],
        {"provenance": IdentityBindingValue(ref="dc/base/ignored")},
        deadline=None,
    )
    assert set(result) == {"provenance"}
    assert result["provenance"].predicate(_facet(provenance_id="dc/base/allowed"))
    assert not result["provenance"].predicate(_facet(provenance_id="dc/base/ignored"))


def test_facet_filter_from_bindings_value_and_set_forms():
    """A single value admits its ref; a set admits each member."""
    result = _facet_filter_from_bindings(
        _GRAPH,
        [],
        {
            "measurementMethod": IdentityBindingValue(ref="GrantEquivalent"),
            "unit": IdentityBindingSet(refs=["USD", "EUR"]),
        },
        deadline=None,
    )
    assert set(result) == {"measurementMethod", "unit"}
    assert result["measurementMethod"].predicate(_facet(measurement_method="GrantEquivalent"))
    assert not result["measurementMethod"].predicate(_facet(measurement_method="Commitment"))
    assert result["unit"].predicate(_facet(unit="USD"))
    assert result["unit"].predicate(_facet(unit="EUR"))
    assert not result["unit"].predicate(_facet(unit="GBP"))


def test_facet_filter_from_bindings_unit_request_form_filters_by_satisfies():
    """A USD/current request matches facets whose unit maps to USD, and no facet
    without a unit."""
    result = _facet_filter_from_bindings(
        _GRAPH,
        [],
        {
            "unit": IdentityFacetValueRequest(
                request=IdentityUnitRequest(currency="USD", prices="current")
            )
        },
        deadline=None,
    )
    assert set(result) == {"unit"}
    assert result["unit"].predicate(_facet(unit="USD"))
    assert not result["unit"].predicate(_facet(unit="EUR"))
    assert not result["unit"].predicate(_facet(unit=None))


def test_facet_filter_from_bindings_unit_request_form_unrecognised_unit_never_satisfies():
    """An unmapped facet unit fails a valid request-form binding."""
    result = _facet_filter_from_bindings(
        _GRAPH,
        [],
        {
            "unit": IdentityFacetValueRequest(
                request=IdentityUnitRequest(currency="USD", prices="current")
            )
        },
        deadline=None,
    )
    assert not result["unit"].predicate(_facet(unit="dc/unknown/unit"))


def test_facet_filter_from_bindings_unit_request_form_never_honours_constant_prices():
    """A native unit ref cannot satisfy a constant-price request."""
    result = _facet_filter_from_bindings(
        _GRAPH,
        [],
        {
            "unit": IdentityFacetValueRequest(
                request=IdentityUnitRequest(currency="USD", prices="constant", base_year=2023)
            )
        },
        deadline=None,
    )
    assert not result["unit"].predicate(_facet(unit="USD"))
