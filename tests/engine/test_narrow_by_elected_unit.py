"""Tests that `narrow_by_elected_unit` filters each StatVar's facets by its own
elected unit refs, and that `FacetAdmission` narrows a bulk plan's role candidates the
same way.
"""

from __future__ import annotations

import pytest

from search.engine.clients.graph import Facet
from search.engine.region.facets import (
    FacetAdmission,
    FacetsBySvRequiredError,
    narrow_by_elected_unit,
)
from search.engine.retrieval.facet_election import merge_role_candidates
from search.engine.retrieval.read_plan import BulkPlan
from tests.engine._harness import lead_selected
from tests.engine._role_candidates import role_candidate

_KEN = (("recipient", "country/KEN"),)


def _facet(facet_id: str, unit: str) -> Facet:
    return Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=1,
        dates=["2020"],
        values=[1.0],
        unit=unit,
        facet_id=facet_id,
    )


# ---------------------------------------------------------------------------
# facets_by_sv
# ---------------------------------------------------------------------------


def test_two_stat_vars_each_keep_only_their_own_elected_unit():
    """Both StatVars share F1 under different units. `sv_a` keeps USD F1 over
    EUR F2; `sv_b` keeps EUR F1 over GBP F3.
    """
    facets_by_sv = {
        "sv_a": [_facet("F1", "USD"), _facet("F2", "EUR")],
        "sv_b": [_facet("F1", "EUR"), _facet("F3", "GBP")],
    }
    elected_units = {"sv_a": frozenset({"USD"}), "sv_b": frozenset({"EUR"})}

    result = narrow_by_elected_unit(facets_by_sv, elected_units)

    assert [f.facet_id for f in result["sv_a"]] == ["F1"]
    assert result["sv_a"][0].unit == "USD"
    assert [f.facet_id for f in result["sv_b"]] == ["F1"]
    assert result["sv_b"][0].unit == "EUR"


def test_stat_var_absent_from_elected_units_keeps_nothing():
    facets_by_sv = {"sv_c": [_facet("F9", "USD")]}

    result = narrow_by_elected_unit(facets_by_sv, {})

    assert result == {"sv_c": []}


def test_facets_by_sv_none_raises():
    with pytest.raises(FacetsBySvRequiredError):
        narrow_by_elected_unit(None, {"sv_a": frozenset({"USD"})})


# ---------------------------------------------------------------------------
# role candidates of a bulk plan
# ---------------------------------------------------------------------------


def _candidate(facet_id: str, unit: str):
    return role_candidate(
        facet_id=facet_id,
        role_tuples=frozenset({_KEN}),
        latest_date="2020",
        provenance_id="dc/base/Source",
        unit=unit,
    )


def test_role_candidates_narrow_per_stat_var():
    """Both StatVars hold USD `F1` and EUR `F2`. `sv_a` elects USD and `sv_b` EUR, so `F1`
    keeps `sv_a` alone and `F2` keeps `sv_b` alone: `sv_a` drops its EUR candidate although
    `sv_b` elects EUR."""
    index = {sv: [_candidate("F1", "USD"), _candidate("F2", "EUR")] for sv in ("sv_a", "sv_b")}
    plan = BulkPlan(
        members=("sv_a", "sv_b"),
        years=None,
        facet_id="F1",
        selected=lead_selected(merge_role_candidates(index, ["sv_a", "sv_b"]), "F1"),
        candidates=merge_role_candidates(index, ["sv_a", "sv_b"]),
    )
    admission = FacetAdmission(
        elected_units={"sv_a": frozenset({"USD"}), "sv_b": frozenset({"EUR"})}
    )

    narrowed = plan.narrowed(admission.admits)

    assert {facet_id: list(c.by_sv) for facet_id, c in narrowed.candidates.items()} == {
        "F1": ["sv_a"],
        "F2": ["sv_b"],
    }


def test_a_role_candidate_whose_unit_no_stat_var_elects_is_dropped():
    index = {"sv_a": [_candidate("F1", "GBP")]}
    plan = BulkPlan(
        members=("sv_a",),
        years=None,
        facet_id="F1",
        selected=lead_selected(merge_role_candidates(index, ["sv_a"]), "F1"),
        candidates=merge_role_candidates(index, ["sv_a"]),
    )

    narrowed = plan.narrowed(FacetAdmission(elected_units={"sv_a": frozenset({"USD"})}).admits)

    assert narrowed.candidates == {}
    assert narrowed.facet_id is None
