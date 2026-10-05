"""Tests for the facet maps shared by the scope and role bulk observation builders."""

from __future__ import annotations

from dataclasses import dataclass

from origin_catalogue import BASE
from search.engine.region.facets import _bulk_candidate_facet_map
from search.models import DataCommonsRef, SeriesFacetRef, SourceRef
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue


@dataclass
class _Candidate:
    """Provide the three attributes shared by bulk facet candidates."""

    provenance_id: str | None
    measurement_method: str | None
    unit: str | None


def test_facet_map_covers_all_three_axes_in_facet_axis_order():
    candidate = _Candidate(
        provenance_id="dc/base/Src", measurement_method="dcAggregate/Census", unit="Person"
    )
    labels = {
        "dc/base/Src": "Source",
        "dcAggregate/Census": "Census",
        "Person": "Person (label)",
    }

    facet_map = _bulk_candidate_facet_map(candidate, labels)

    # Preserve `FacetAxis` order in the serialized payload.
    assert list(facet_map) == ["provenance", "measurementMethod", "unit"]
    assert facet_map == {
        "provenance": SeriesFacetRef(ref="dc/base/Src", label="Source"),
        "measurementMethod": SeriesFacetRef(ref="dcAggregate/Census", label="Census"),
        "unit": SeriesFacetRef(ref="Person", label="Person (label)"),
    }


def test_facet_map_omits_axes_with_no_value():
    candidate = _Candidate(provenance_id="dc/base/Src", measurement_method=None, unit=None)

    facet_map = _bulk_candidate_facet_map(candidate, {"dc/base/Src": "Source"})

    assert facet_map == {"provenance": SeriesFacetRef(ref="dc/base/Src", label="Source")}


def test_facet_map_provenance_prefers_the_catalogue_row(bind_catalogue):
    bind_catalogue(
        restrictive_catalogue(provenances=[provenance_row("dc/base/Src", BASE, source="dc/s/Pub")])
    )
    candidate = _Candidate(provenance_id="dc/base/Src", measurement_method=None, unit=None)

    facet_map = _bulk_candidate_facet_map(candidate, {"dc/base/Src": "Graph label"})

    assert facet_map == {
        "provenance": SeriesFacetRef(
            ref="dc/base/Src",
            label="dc/base/Src",
            source=SourceRef(ref="dc/s/Pub", label="dc/s/Pub"),
            data_commons=DataCommonsRef(id="base", label="Data Commons by Google"),
        )
    }


def test_facet_map_ref_with_no_label_stays_unlabelled():
    candidate = _Candidate(provenance_id=None, measurement_method="dcAggregate/Census", unit=None)

    facet_map = _bulk_candidate_facet_map(candidate, {})

    assert facet_map == {"measurementMethod": SeriesFacetRef(ref="dcAggregate/Census", label=None)}
