"""A same-shape merge rebuilds its read plan without a graph read
(`merged_plan.merged_read_plan`), and only where its members' plans decide it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from materialise.facets import bound_variable_breadth
from origin_catalogue import current_catalogue
from search.engine import same_shape
from search.engine.clients.graph import Facet
from search.engine.region.composite_answer import ComponentReads
from search.engine.region.result import RegionResult
from search.engine.retrieval.coverage import CoverageForm
from search.engine.retrieval.merged_plan import merged_read_plan
from search.engine.retrieval.read_plan import PairPlan, plan_standard
from search.models import Spec
from tests.conftest import minimal_spec
from tests.engine._fake_coverage import build_coverage_index, empty_coverage_snapshot

_KEN = "country/KEN"


def _facet(facet_id: str, obs_count: int) -> Facet:
    return Facet(
        earliest_date="2010",
        latest_date="2020",
        obs_count=obs_count,
        dates=[],
        provenance_id=f"dc/base/{facet_id}",
        facet_id=facet_id,
    )


@pytest.mark.parametrize(("counts", "ties"), [((5, 5), True), ((5, 3), False)])
def test_a_standard_merge_at_one_place_reads_no_breadth_to_break_a_tie(counts, ties):
    """Two variables at one place, each held by one facet of its own source. Merged, they
    elect one group across both. Where the two facets hold the same points, only the variables' whole
    breadth breaks the tie, which a graph read settles, so the merge rebuilds no plan and
    the request's breadth reader is never called. Where they differ, the plan is rebuilt."""
    reads: list[Sequence[str]] = []

    def reader(stat_vars: Sequence[str], *, deadline: float | None) -> dict:
        reads.append(stat_vars)
        return {}

    with bound_variable_breadth(reader, refuse_unread=False):
        plans = [
            plan_standard(
                {(sv, _KEN): [_facet(f"f_{sv}", count)]}, members=[sv], years=None, places=[_KEN]
            )
            for sv, count in zip(("sv_a", "sv_b"), counts, strict=True)
        ]
        merged = merged_read_plan(plans, coverage_index=build_coverage_index([], role_names={}))

    assert reads == []
    if ties:
        assert merged is None
    else:
        assert isinstance(merged, PairPlan)
        assert merged.members == ("sv_a", "sv_b")


@pytest.mark.parametrize("omitted", [False, True])
def test_a_standard_merge_with_rows_a_conversion_left_out_rebuilds_no_plan(omitted):
    """A conversion leaves a member's row out after its group was elected over it
    (`PairPlan.omitting`), so the member's plan no longer holds what the merged election
    would read. The merge rebuilds no plan then, and does otherwise."""
    plans = [
        plan_standard({(sv, _KEN): [_facet(f"f_{sv}", count)]}, members=[sv], years=None)
        for sv, count in (("sv_a", 5), ("sv_b", 3))
    ]
    if omitted:
        plans[0] = plans[0].omitting(frozenset({("sv_a", "observationAbout", _KEN)}))

    merged = merged_read_plan(plans, coverage_index=build_coverage_index([], role_names={}))

    assert (merged is None) is omitted


def _member(name: str, plan: PairPlan, *, units_elected: bool) -> RegionResult:
    return RegionResult(
        variable_text=name,
        status="definite",
        specs=(Spec.model_validate(minimal_spec(name)),),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        served_plans=(plan,),
        coverage_form=CoverageForm(),
        units_elected=units_elected,
    )


@pytest.mark.parametrize("units_elected", [False, True])
def test_a_merge_of_answers_narrowed_to_their_own_elected_units_rebuilds_no_plan(units_elected):
    """A unit request elects each answer's units over its own group, which a resolve of the
    merged identity elects over the merged group, so the merge rebuilds no plan for members
    narrowed by them, and does for members that were not."""
    group = [
        _member(
            sv,
            plan_standard({(sv, _KEN): [_facet(f"f_{sv}", count)]}, members=[sv], years=None),
            units_elected=units_elected and sv == "sv_b",
        )
        for sv, count in (("sv_a", 5), ("sv_b", 3))
    ]
    reads = ComponentReads(
        graph=cast(Any, None),
        sdmx=cast(Any, None),
        coverage=empty_coverage_snapshot(),
        materialise_graph=None,
        materialise_sdmx=None,
        rate_cache=None,
        deadline=None,
        catalogue=current_catalogue(),
    )

    counted = same_shape._merged_counts(
        group, group[0].spec, diff_key=("how", "sector"), reads=reads
    )

    assert (counted is None) is units_elected
