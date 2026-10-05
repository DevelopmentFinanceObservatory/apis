"""A posted facet binding narrows a bulk plan's candidates by their shared facet metadata."""

from __future__ import annotations

from dataclasses import replace

from search.engine.clients.sdmx import RoleFacetCandidate
from search.engine.region.facets import FacetAdmission
from search.engine.region.resubmit_request import _facet_filter_from_bindings
from search.engine.retrieval.coverage import CoverageForm, answer_coverage
from search.engine.retrieval.facet_election import merge_role_candidates
from search.engine.retrieval.read_plan import BulkPlan
from search.models import (
    BreadthDim,
    CoverageBreadth,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetValueAbsent,
    TimeWindow,
)
from tests.engine._role_candidates import role_candidate
from tests.fixtures import FakeGraph

_KEN = (("recipient", "country/KEN"),)
_NGA = (("recipient", "country/NGA"),)
_TZA = (("recipient", "country/TZA"),)


def _usd_candidate() -> RoleFacetCandidate:
    """Facet F1: USD, Census, latest 2022, over KEN and NGA."""
    return role_candidate(
        facet_id="F1",
        role_tuples=frozenset({_KEN, _NGA}),
        latest_date="2022",
        provenance_id="dc/base/UsdSource",
        measurement_method="dcAggregate/Census",
        unit="USD",
    )


def _eur_candidate() -> RoleFacetCandidate:
    """Facet F2: EUR, Estimate, latest 2023 (the overall latest), over TZA."""
    return role_candidate(
        facet_id="F2",
        role_tuples=frozenset({_TZA}),
        latest_date="2023",
        provenance_id="dc/base/EurSource",
        measurement_method="dcAggregate/Estimate",
        unit="EUR",
    )


def _role_index() -> dict[str, list[RoleFacetCandidate]]:
    return {"sv_a": [_usd_candidate(), _eur_candidate()]}


def _served_facet_after(bindings, candidates: list[RoleFacetCandidate]) -> str | None:
    """The facet a bulk plan over `candidates` serves once the posted `bindings` narrow it
    (`resubmit_request._facet_filter_from_bindings`, `facets.FacetAdmission`)."""
    merged = merge_role_candidates({"sv_a": candidates}, ["sv_a"])
    plan = BulkPlan(members=("sv_a",), years=None, facet_id=None, candidates=merged)
    facet_filter = _facet_filter_from_bindings(FakeGraph(), [], bindings, deadline=None)
    return plan.narrowed(FacetAdmission(axes=facet_filter).admits).facet_id


def test_multiple_axes_combine_as_a_conjunction():
    """F1 and F2 each fail one of the two bound axes."""
    bindings = {
        "unit": IdentityBindingValue(ref="USD"),
        "measurementMethod": IdentityBindingValue(ref="dcAggregate/Estimate"),
    }
    assert _served_facet_after(bindings, _role_index()["sv_a"]) is None


def test_multiple_axes_admit_the_candidate_satisfying_both():
    bindings = {
        "unit": IdentityBindingValue(ref="USD"),
        "measurementMethod": IdentityBindingValue(ref="dcAggregate/Census"),
    }
    assert _served_facet_after(bindings, _role_index()["sv_a"]) == "F1"


def test_candidate_dropped_when_a_bound_axis_has_no_value():
    """A missing measurement_method fails every posted ref and set binding: the predicate
    `_facet_binding_admits` builds receives `None` and admits only declared values."""
    candidate = replace(_usd_candidate(), measurement_method=None)
    for binding in (
        IdentityBindingValue(ref="dcAggregate/Census"),
        IdentityBindingSet(refs=["dcAggregate/Census", "dcAggregate/Estimate"]),
    ):
        assert _served_facet_after({"measurementMethod": binding}, [candidate]) is None


def test_null_ref_binding_keeps_only_candidates_without_the_axis():
    """A posted null ref admits the candidate that declares no measurement method."""
    no_method = replace(_eur_candidate(), measurement_method=None)
    bindings = {"measurementMethod": IdentityFacetValueAbsent(ref=None)}
    assert _served_facet_after(bindings, [_usd_candidate(), no_method]) == "F2"


# A discovered bulk plan narrowed by a unit filter states coverage from what it still serves.


_DISCOVERED = CoverageForm(variables=True, open_axis=("recipients", "recipient"))


def _discovered_plan(years) -> BulkPlan:
    """The bulk plan discovery records over the unfiltered index: `F1` covers two pairs to
    `F2`'s one, so it is elected."""
    merged = merge_role_candidates(_role_index(), ["sv_a"])
    return BulkPlan(members=("sv_a",), years=years, facet_id="F1", candidates=merged)


def _eur(_sv: str, candidate) -> bool:
    return candidate.unit == "EUR"


def test_a_unit_filter_reelects_and_recounts_the_served_recipients():
    """EUR drops `F1`; the narrowed plan serves `F2` at Tanzania alone, one source."""
    narrowed = _discovered_plan(None).narrowed(_eur)

    coverage = answer_coverage(narrowed, _DISCOVERED)

    assert narrowed.facet_id == "F2"
    assert isinstance(coverage, CoverageBreadth)
    assert coverage.dimensions == [
        BreadthDim(label="recipients", count=1),
        BreadthDim(label="sources", count=1),
        BreadthDim(label="variables", count=1),
    ]


def test_a_unit_filter_under_latest_dates_the_reelected_facets_year():
    """Unfiltered, `F1` dates the answer 2022; under EUR `F2` serves and dates it 2023."""
    plan = _discovered_plan("latest")

    assert answer_coverage(plan, _DISCOVERED).window == TimeWindow(start_year=2022, end_year=2022)
    assert answer_coverage(plan.narrowed(_eur), _DISCOVERED).window == TimeWindow(
        start_year=2023, end_year=2023
    )


def test_a_filter_admitting_nothing_serves_nothing():
    narrowed = _discovered_plan(None).narrowed(lambda _sv, _c: False)

    coverage = answer_coverage(narrowed, _DISCOVERED)

    assert narrowed.facet_id is None
    assert coverage.has_data is False
    assert [d.count for d in coverage.dimensions] == [0, 0, 0]
